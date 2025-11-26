from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence
from retrying import retry

import torch
import torch.distributed as dist
import wandb
from PIL import Image
from tqdm import trange
from qwen_vl_utils import process_vision_info

from veomni.checkpoint import build_checkpointer, ckpt_to_state_dict
from veomni.data import (
    OmniDataCollatorWithPacking,
    OmniDataCollatorWithPadding,
    OmniSequenceShardCollator,
    build_dataloader,
    build_interleave_dataset,
    build_iterative_dataset,
    build_mapping_dataset,
    build_multimodal_chat_template,
)
from veomni.data.constants import IMAGE_INPUT_INDEX
from veomni.data.data_collator import DataCollator
from veomni.data.multimodal.preprocess import conv_preprocess
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, build_processor, save_model_assets, save_model_weights
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args, save_args
from veomni.utils.device import (
    get_device_type,
    get_nccl_backend,
    get_torch_device,
    synchronize,
)
from veomni.utils.dist_utils import all_reduce
from qwen_vl_utils import fetch_image


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from veomni.data.chat_template import ChatTemplate


logger = helper.create_logger(__name__)


MAX_PIXELS = 256 * 28 * 28
ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
}


@retry(stop_max_attempt_number=10)
def fetch_image_fn(image_url: str):
    return fetch_image(
        {
            "image_url": image_url,
            "max_pixels": MAX_PIXELS,
        }
    )


@dataclass
class BatchTransformSampleDataCollator(DataCollator):

    def __init__(self, processor: "ProcessorMixin", position_id_func: "Callable", max_seq_len: int):
        self.thread_pool = ThreadPoolExecutor(max_workers=100)
        self.processor = processor
        self.position_id_func = position_id_func
        self.max_seq_len = max_seq_len
        self.truncate_count = 0

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, "torch.Tensor"]]:
        image_urls: list[str] = [image_url for sample in batch if sample.get("image_urls") for image_url in sample.get("image_urls")]
        images = self.thread_pool.map(fetch_image_fn, image_urls) if len(image_urls) > 0 else []
        image_map = {image_url: image for image_url, image in zip(image_urls, images)} if len(image_urls) > 0 else {}

        tokenized_examples = []
        for sample in batch:
            input_ids = sample["input_ids"]
            labels = sample.get("labels", input_ids)
            image_urls = sample.get("image_urls", [])

            image_inputs = {}
            image_grid_thw = None

            original_len = len(input_ids)

            if image_urls:
                images = [image_map[image_url] for image_url in image_urls]
                image_inputs = self.processor.image_processor(images=images, return_tensors="pt")
                image_grid_thw = image_inputs.get("image_grid_thw")

                merge_length = self.processor.image_processor.merge_size**2
                image_token_nums = image_grid_thw.prod(dim=-1) // merge_length

                input_ids_tensor = torch.tensor(input_ids)
                labels_tensor = torch.tensor(labels)
                image_mask = input_ids_tensor == self.processor.image_token_id
                image_indices = torch.where(image_mask)[0].tolist()

                expanded_input_ids = []
                expanded_labels = []
                prev_end = 0
                valid_image_indices = []

                # 计算非图像token数：总token数 - 图像token数（每个图像占3个token：vision_start_token + image_token + vision_end_token）
                keep_token_nums = len(input_ids_tensor) - len(image_indices) * 3

                for image_idx, img_pos in enumerate(image_indices):
                    # 图像结构：vision_start_token image_token vision_end_token
                    # 默认 image_token 前一个就是 vision_start_token，后一个就是 vision_end_token
                    vision_start_pos = img_pos - 1
                    vision_end_pos = img_pos + 1
                    
                    # 添加 vision_start_token 之前的文本
                    text_before = input_ids_tensor[prev_end:vision_start_pos]
                    expanded_input_ids.append(text_before)
                    expanded_labels.append(labels_tensor[prev_end:vision_start_pos])

                    # 合并添加：vision_start_token + 展开的 image_token + vision_end_token
                    num_tokens = image_token_nums[image_idx].item()
                    image_tokens = torch.full((num_tokens,), self.processor.image_token_id, dtype=torch.long)
                    image_segment = torch.cat([
                        input_ids_tensor[vision_start_pos:vision_start_pos+1],  # vision_start_token
                        image_tokens,  # 展开的 image_token
                        input_ids_tensor[vision_end_pos:vision_end_pos+1],  # vision_end_token
                    ])
                    image_labels_segment = torch.cat([
                        labels_tensor[vision_start_pos:vision_start_pos+1],  # vision_start_token label
                        labels_tensor[img_pos : img_pos + 1].expand(num_tokens),  # 展开的 image_token labels
                        labels_tensor[vision_end_pos:vision_end_pos+1],  # vision_end_token label
                    ])
                    image_segment_len = len(image_segment)
                    if keep_token_nums + image_segment_len <= self.max_seq_len:
                        expanded_input_ids.append(image_segment)
                        expanded_labels.append(image_labels_segment)
                        keep_token_nums += image_segment_len
                        valid_image_indices.append(image_idx)
                    else:
                        pass
                    original_len += image_segment_len - 3

                    prev_end = vision_end_pos + 1

                expanded_input_ids.append(input_ids_tensor[prev_end:])
                expanded_labels.append(labels_tensor[prev_end:])

                input_ids = torch.cat(expanded_input_ids).tolist()
                labels = torch.cat(expanded_labels).tolist()

                # 只保留有效图像的 grid_thw 和相关数据
                if len(valid_image_indices) < len(image_token_nums):
                    # 保存原始的 image_grid_thw，用于计算每个图像的序列长度
                    original_image_grid_thw = image_grid_thw
                    image_grid_thw = image_grid_thw[valid_image_indices]
                    # 同时需要更新 image_inputs 中的其他字段
                    if "pixel_values" in image_inputs:
                        # pixel_values 的第一维是所有图像的序列长度之和，需要根据每个图像的序列长度来分割
                        # 计算每个图像的序列长度（prod 得到的是特征数量，即序列长度）
                        image_seq_lengths = original_image_grid_thw.prod(dim=-1)
                        # 根据序列长度分割 pixel_values
                        pixel_values_list = torch.split(image_inputs["pixel_values"], image_seq_lengths.tolist())
                        # 只保留有效图像对应的 pixel_values
                        valid_pixel_values = [pixel_values_list[idx] for idx in valid_image_indices]
                        # 重新拼接
                        image_inputs["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
                    if "image_grid_thw" in image_inputs:
                        image_inputs["image_grid_thw"] = image_grid_thw
                    if self.truncate_count == 0:
                        logger.info_rank0(f"样本长度超过最大长度，截断到 {len(input_ids)} 个 token，原始长度为 {original_len}，保留 {len(valid_image_indices)} 个完整图像")
                        self.truncate_count += 1
                    self.truncate_count += 1

            tokenized_example = {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor([1] * len(input_ids)),
                "labels": torch.tensor(labels),
            }

            tokenized_example["image_mask"] = tokenized_example["input_ids"] == self.processor.image_token_id
            # tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
            tokenized_example.update(image_inputs)

            position_ids = self.position_id_func(
                input_ids=tokenized_example["input_ids"].unsqueeze(0),
                image_grid_thw=image_grid_thw,
                attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
            )["position_ids"]
            tokenized_example["position_ids"] = position_ids.squeeze().clone()

            tokenized_examples.append(tokenized_example)

        return tokenized_examples


def process_prepared_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    max_seq_len: int,
    source_name: Optional[str] = None,
    position_id_func: "Callable" = None,
) -> List[Dict[str, "torch.Tensor"]]:
    return [sample]


def process_sample(
    sample: Dict[str, Any],
    processor: "ProcessorMixin",
    chat_template: "ChatTemplate",
    position_id_func: "Callable",
    **kwargs,
):
    """
    Processes multimodal example with qwen2_5_vl's pre-processor.
    """
    source_name = sample["source_name"] if "source_name" in sample else kwargs["source_name"]
    conversations = sample["text"] if source_name == "fineweb_100BT" else sample["conversations"]  # text-only data
    conversations = conv_preprocess(source_name, conversations, **kwargs)

    token_num_inputs, image_inputs = {}, {}
    image_grid_thw = None

    if "images" in sample and sample["images"]:
        images = []
        for image in sample["images"]:
            images.append(Image.open(BytesIO(image)).convert("RGB"))

        image_inputs = processor.image_processor(images=images, return_tensors="pt")
        image_grid_thw = image_inputs["image_grid_thw"]
        merge_length = processor.image_processor.merge_size**2
        image_token_num = image_grid_thw.prod(dim=-1) // merge_length
        token_num_inputs["image"] = image_token_num

    tokenized_example = chat_template.encode_messages(conversations, token_num_inputs)
    tokenized_example = {k: torch.tensor(v) for k, v in tokenized_example.items()}
    input_ids = tokenized_example["input_ids"]

    position_ids = position_id_func(
        input_ids=input_ids.unsqueeze(0),
        image_grid_thw=image_grid_thw,
        attention_mask=tokenized_example["attention_mask"].unsqueeze(0),
    )["position_ids"]
    tokenized_example["position_ids"] = position_ids.squeeze().clone()  # (dim, l)

    tokenized_example["image_mask"] = tokenized_example["input_ids"] == IMAGE_INPUT_INDEX
    tokenized_example["input_ids"][tokenized_example["image_mask"]] = 0
    tokenized_example.update(image_inputs)
    return [tokenized_example]


def get_param_groups(model: "torch.nn.Module", default_lr: float, vit_lr: float):
    vit_params, other_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "visual" in name:
                vit_params.append(param)
            else:
                other_params.append(param)

    return [{"params": vit_params, "lr": vit_lr}, {"params": other_params, "lr": default_lr}]


@dataclass
class MyTrainingArguments(TrainingArguments):
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Whether or not to freeze the vit parameters."},
    )
    vit_lr: float = field(
        default=1e-6,
        metadata={"help": "Maximum learning rate for vit parameters."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "DataArguments" = field(default_factory=DataArguments)
    train: "MyTrainingArguments" = field(default_factory=MyTrainingArguments)


def main():
    args = parse_args(Arguments)
    # logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    # logger.info_rank0(json.dumps(asdict(args), indent=2))
    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    if not dist.is_initialized():
        dist.init_process_group(backend=get_nccl_backend())
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()

    # if args.train.global_rank == 0:
    #     save_args(args, args.train.output_dir)

    Checkpointer = build_checkpointer(dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager)

    init_parallel_state(
        dp_size=args.train.data_parallel_size,
        dp_replicate_size=args.train.data_parallel_replicate_size,
        dp_shard_size=args.train.data_parallel_shard_size,
        tp_size=args.train.tensor_parallel_size,
        ep_size=args.train.expert_parallel_size,
        pp_size=args.train.pipeline_parallel_size,
        cp_size=args.train.context_parallel_size,
        ulysses_size=args.train.ulysses_parallel_size,
        dp_mode=args.train.data_parallel_mode,
    )

    logger.info_rank0("Prepare model")
    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.enable_mixed_precision else "bfloat16",
        init_device=args.train.init_device,
        force_use_huggingface=args.model.force_use_huggingface,
        attn_implementation=args.model.attn_implementation,
    )
    model_config = model.config
    # helper.print_device_mem_info("VRAM usage after building model")

    logger.info_rank0("Prepare data")
    processor = build_processor(args.model.tokenizer_path)
    processor.image_processor.max_pixels = MAX_PIXELS
    position_id_func = model.get_position_id_func()
    processor.image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    if args.data.data_type == "prepared":
        transform = partial(
            process_prepared_sample,
            processor=processor,
            max_seq_len=args.data.max_seq_len,
            position_id_func=position_id_func,
        )
    elif args.data.data_type == "conversation":
        chat_template = build_multimodal_chat_template(args.data.chat_template, processor.tokenizer)
        transform = partial(
            process_sample,
            processor=processor,
            chat_template=chat_template,
            position_id_func=position_id_func,
        )

    if args.train.rmpad:
        raise ValueError("Qwen2-VL does not support rmpad. Use `rmpad_with_pos_ids` instead.")

    data_collate_fn = [BatchTransformSampleDataCollator(processor, position_id_func, args.data.max_seq_len)]
    if args.train.rmpad_with_pos_ids:
        data_collate_fn.append(OmniDataCollatorWithPacking())
    else:
        data_collate_fn.append(OmniDataCollatorWithPadding())
    if get_parallel_state().sp_enabled:
        data_collate_fn.append(
            OmniSequenceShardCollator(
                padding_scale={
                    "pixel_values": processor.image_processor.merge_size**2,
                },
                rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            )
        )

    if args.data.dataloader_type == "native":
        if args.data.enable_multisource:
            logger.info_rank0("Start building interleave dataset")
            train_dataset = build_interleave_dataset(
                args.data.train_path, args.data.datasets_type, transform=transform, seed=args.train.seed
            )
        elif args.data.datasets_type == "iterable":
            logger.info_rank0("Start building iterative dataset")
            train_dataset = build_iterative_dataset(
                args.data.train_path, transform=transform, seed=args.train.seed, source_name=args.data.source_name
            )
        elif args.data.datasets_type == "mapping":
            logger.info_rank0("Start building mapping dataset")
            train_dataset = build_mapping_dataset(
                args.data.train_path, transform=transform, source_name=args.data.source_name
            )

        dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
        if args.data.datasets_type == "mapping":
            dataset_length = dataset_length / args.train.data_parallel_size
        args.train.compute_train_steps(args.data.max_seq_len, args.data.train_size, dataset_length)

        train_dataloader = build_dataloader(
            dataset=train_dataset,
            micro_batch_size=args.train.micro_batch_size,
            global_batch_size=args.train.global_batch_size,
            dataloader_batch_size=args.train.dataloader_batch_size,
            seed=args.train.seed,
            collate_fn=data_collate_fn,
            max_seq_len=args.data.max_seq_len,
            train_steps=args.train.train_steps,
            rmpad=args.train.rmpad,
            rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
            bsz_warmup_ratio=args.train.bsz_warmup_ratio,
            dyn_bsz_margin=args.train.dyn_bsz_margin,
            dyn_bsz_buffer_size=args.train.dyn_bsz_buffer_size,
            num_workers=args.data.num_workers,
            drop_last=args.data.drop_last,
            pin_memory=args.data.pin_memory,
            prefetch_factor=args.data.prefetch_factor,
        )
    else:
        raise NotImplementedError(f"Unsupported dataloader type: {args.data.dataloader_type}.")

    fsdp_kwargs = {}
    if args.train.freeze_vit:
        model.visual.requires_grad_(False)
        if args.train.data_parallel_mode == "fsdp1":
            fsdp_kwargs["use_orig_params"] = True

    model = build_parallelize_model(
        model,
        weights_path=args.model.model_path,
        enable_full_shard=args.train.enable_full_shard,
        enable_mixed_precision=args.train.enable_mixed_precision,
        enable_gradient_checkpointing=args.train.enable_gradient_checkpointing,
        init_device=args.train.init_device,
        enable_fsdp_offload=args.train.enable_fsdp_offload,
        fsdp_kwargs=fsdp_kwargs,
        basic_modules=model._no_split_modules,
        enable_reentrant=args.train.enable_reentrant,
        enable_forward_prefetch=args.train.enable_forward_prefetch,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
    )
    optimizer = build_optimizer(
        model,
        lr=args.train.lr,
        weight_decay=args.train.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer,
        param_groups=get_param_groups(model, args.train.lr, args.train.vit_lr),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train.train_steps * args.train.num_train_epochs,
        lr=args.train.lr,
        lr_min=args.train.lr_min,
        lr_decay_style=args.train.lr_decay_style,
        lr_decay_ratio=args.train.lr_decay_ratio,
        lr_warmup_ratio=args.train.lr_warmup_ratio,
        lr_start=args.train.lr_start,
    )

    # if args.train.global_rank == 0:
    #     if args.train.use_wandb:
    #         wandb.init(
    #             project=args.train.wandb_project,
    #             name=args.train.wandb_name,
    #             config={**vars(args.model), **vars(args.data), **vars(args.train)},  # flatten dict
    #         )

    #     model_assets = [model_config, processor]
    #     save_model_assets(args.train.model_assets_dir, model_assets)

    if args.train.profile_this_rank:
        profiler = helper.create_profiler(
            start_step=args.train.profile_start_step,
            end_step=args.train.profile_end_step,
            trace_dir=args.train.profile_trace_dir,
            record_shapes=args.train.profile_record_shapes,
            profile_memory=args.train.profile_profile_memory,
            with_stack=args.train.profile_with_stack,
            global_rank=args.train.global_rank,
        )
        profiler.start()

    start_epoch, start_step, global_step = 0, 0, 0
    save_checkpoint_path = None
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        rmpad=args.train.rmpad,
        rmpad_with_pos_ids=args.train.rmpad_with_pos_ids,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
        empty_cache_steps=args.train.empty_cache_steps,
    )

    if args.train.load_checkpoint_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}  # cannot be None
        Checkpointer.load(args.train.load_checkpoint_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train.train_steps
        start_step = global_step % args.train.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        if start_step == 0:  # resume at the end of epoch
            iter(train_dataloader)  # clear resume state and prefetch data

        if args.train.global_rank == 0:
            helper.load_step2token(args.train.load_checkpoint_path)
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    helper.empty_cache()
    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.enable_activation_offload, args.train.enable_gradient_checkpointing, args.train.activation_gpu_limit
    )
    model.train()
    logger.info_rank0("Start training")
    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)

        # data_loader_tqdm = trange(
        #     args.train.train_steps,
        #     desc=f"Epoch {epoch + 1}/{args.train.num_train_epochs}",
        #     total=args.train.train_steps,
        #     initial=start_step,
        #     disable=args.train.local_rank != 0,
        # )
        data_iterator = iter(train_dataloader)
        for _ in range(start_step, args.train.train_steps):
            global_step += 1
            try:
                micro_batches: List[Dict[str, Any]] = next(data_iterator)
            except StopIteration:
                logger.info(f"epoch:{epoch} Dataloader finished with drop_last {args.data.drop_last}")
                break

            if global_step == 1:
                helper.print_example(example=micro_batches[0], rank=args.train.local_rank)

            total_loss = 0
            synchronize()
            start_time = time.time()
            for micro_batch in micro_batches:
                environ_meter.add(micro_batch)
                if args.data.enable_multisource:
                    micro_batch.pop("ds_idx", None)
                    micro_batch.pop("source_name", None)

                micro_batch = {
                    k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in micro_batch.items()
                }
                with model_fwd_context:
                    loss: "torch.Tensor" = model(**micro_batch, use_cache=False).loss / len(micro_batches)

                with model_bwd_context:
                    loss.backward()

                total_loss += loss.item()
                del micro_batch

            if args.train.data_parallel_mode == "fsdp1":
                grad_norm = model.clip_grad_norm_(args.train.max_grad_norm).item()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.train.max_grad_norm, foreach=True)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()

            # collect mean loss across data parallel group
            total_loss, grad_norm = all_reduce((total_loss, grad_norm), group=get_parallel_state().fsdp_group)
            synchronize()
            delta_time = time.time() - start_time
            lr = max(lr_scheduler.get_last_lr())
            train_metrics = environ_meter.step(delta_time, global_step=global_step)

            # data_loader_tqdm.set_postfix_str(f"loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}")
            # data_loader_tqdm.update()
            logger.info_rank0(
                f"epoch: {epoch + 1}, step: {global_step} / {args.train.train_steps}, loss: {total_loss:.2f}, grad_norm: {grad_norm:.2f}, lr: {lr:.2e}, elapsed time: {delta_time:.2f}s"
            )

            if args.train.global_rank == 0:
                if args.train.use_wandb:
                    train_metrics.update(
                        {"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": lr}
                    )
                    wandb.log(train_metrics, step=global_step)

            if args.train.profile_this_rank and global_step <= args.train.profile_end_step:
                profiler.step()
                if global_step == args.train.profile_end_step:
                    profiler.stop()

            if args.train.save_steps and global_step % args.train.save_steps == 0:
                helper.empty_cache()
                save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                    },
                }
                Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
                # if args.train.global_rank == 0:
                #     helper.save_step2token(
                #         args.train.step2token_path,
                #         consumed_tokens=train_metrics["consume_tokens(B)"],
                #         global_step=global_step,
                #         save_checkpoint_path=save_checkpoint_path,
                #     )
                dist.barrier()
                logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

        # data_loader_tqdm.close()
        start_step = 0
        # helper.print_device_mem_info(f"VRAM usage after epoch {epoch + 1}")
        if args.train.save_epochs and (epoch + 1) % args.train.save_epochs == 0:
            helper.empty_cache()
            save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{global_step}")
            state = {
                "model": model,
                "optimizer": optimizer,
                "extra_state": {
                    "global_step": global_step,
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "train_dataloader": train_dataloader.state_dict(),
                    "environ_meter": environ_meter.state_dict(),
                },
            }
            Checkpointer.save(args.train.save_checkpoint_path, state, global_steps=global_step)
            # if args.train.global_rank == 0:
            #     helper.save_step2token(
            #         args.train.step2token_path,
            #         consumed_tokens=train_metrics["consume_tokens(B)"],
            #         global_step=global_step,
            #         save_checkpoint_path=save_checkpoint_path,
            #     )
            dist.barrier()
            logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    helper.empty_cache()
    state = {
        "model": model,
        "optimizer": optimizer,
        "extra_state": {
            "global_step": global_step,
            "lr_scheduler": lr_scheduler.state_dict(),
            "train_dataloader": train_dataloader.state_dict(),
            "environ_meter": environ_meter.state_dict(),
        },
    }
    logger.info_rank0(f"saving final checkpoint to {args.train.final_checkpoint_path} ...")
    Checkpointer.save(args.train.final_checkpoint_path, state)
    # logger.info_rank0(f"final checkpoint saved at {args.train.final_checkpoint_path} successfully!")
    dist.barrier()

    synchronize()
    # release memory
    del optimizer, lr_scheduler
    helper.empty_cache()
    # save model in huggingface's format
    if args.train.global_rank == 0:
        # if args.train.save_hf_weights and save_checkpoint_path is not None:
        #     hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        #     model_state_dict = ckpt_to_state_dict(
        #         save_checkpoint_path=save_checkpoint_path,
        #         output_dir=args.train.output_dir,
        #         ckpt_manager=args.train.ckpt_manager,
        #     )
        #     save_model_weights(hf_weights_path, model_state_dict, model_assets=model_assets)
        #     logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")
        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=args.train.final_checkpoint_path,
            # output_dir=args.train.output_dir,
            ckpt_manager=args.train.ckpt_manager,
        )
        model_assets = [model_config, processor]
        logger.info_rank0(f"saving final model to {args.train.output_model_dir} ...")
        save_model_weights(args.train.output_model_dir, model_state_dict, model_assets=model_assets)
        # logger.info_rank0(f"final model saved at {args.train.output_model_dir} successfully!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
