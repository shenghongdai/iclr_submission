#!/usr/bin/env python3
"""
Fine-tune Gemma 3 (or compatible chat models) directly from a JSON dataset
containing pre-built chat messages. Supports optional images per sample
via `image_path` or embedded in messages, without adding or managing any special/control tokens.

Expected JSON format (a list of dicts):
[
  {
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": [{"type": "text", "text": "..."}, {"type": "image", "path": "/path/to/image.jpg"}]},
      {"role": "assistant", "content": "..."}
    ],
    "image_path": "/path/to/image.jpg"  # optional legacy format; if missing or invalid, trains text-only
  },
  ...
]

The script supports two image formats:
1. Legacy: single image_path per sample
2. New: images embedded in message content as {"type": "image", "path": "..."}
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
from PIL import Image, ImageFile
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer
from accelerate import Accelerator

# XPU helper (compatible with any transformers version)
try:  # >=4.38
    from transformers.utils import is_torch_xpu_available as is_xpu_available
except ImportError:
    try:  # <=4.37
        from transformers import is_xpu_available  # type: ignore
    except ImportError:
        def is_xpu_available() -> bool:  # noqa: D401
            return False

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Gemma3ForConditionalGeneration,
    TrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)

# Allow truncated images without raising
ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_json_dataset(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("JSON root must be a list of samples")
    # Basic validation
    for i, ex in enumerate(data[:5]):
        if "messages" not in ex or not isinstance(ex["messages"], list):
            raise ValueError(f"Sample {i} missing 'messages' list")
    return data


def convert_legacy_to_new_format(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert legacy format (image_path) to new format (images in messages)"""
    converted_data = []
    
    for sample in data:
        new_sample = {"messages": []}
        
        # Copy messages, ensuring consistent content format
        for msg in sample["messages"]:
            new_msg = {"role": msg["role"]}
            
            # Handle content conversion
            if msg["role"] == "user" and "image_path" in sample and sample["image_path"]:
                # Convert user message to include image
                content = []
                if isinstance(msg["content"], str):
                    content.append({"type": "text", "text": msg["content"]})
                elif isinstance(msg["content"], list):
                    content.extend(msg["content"])
                else:
                    # Fallback: wrap any other content type in text
                    content.append({"type": "text", "text": str(msg["content"])})
                
                # Add image
                content.append({"type": "image", "path": sample["image_path"]})
                new_msg["content"] = content
            else:
                # For non-user messages or messages without images, ensure content is a list
                if isinstance(msg["content"], str):
                    new_msg["content"] = [{"type": "text", "text": msg["content"]}]
                elif isinstance(msg["content"], list):
                    new_msg["content"] = msg["content"]
                else:
                    # Fallback: wrap any other content type in text
                    new_msg["content"] = [{"type": "text", "text": str(msg["content"])}]
            
            new_sample["messages"].append(new_msg)
        
        converted_data.append(new_sample)
    
    return converted_data


def validate_converted_data(data: List[Dict[str, Any]], max_samples: int = 5) -> None:
    """Validate that converted data has consistent structure"""
    print(f"\nValidating converted data structure (checking first {max_samples} samples)...")
    
    for i, sample in enumerate(data[:max_samples]):
        print(f"\nSample {i+1}:")
        print(f"  Keys: {list(sample.keys())}")
        
        if "messages" not in sample:
            print(f"  ERROR: Missing 'messages' key")
            continue
            
        for j, msg in enumerate(sample["messages"]):
            print(f"  Message {j+1} ({msg.get('role', 'unknown')}):")
            
            if "content" not in msg:
                print(f"    ERROR: Missing 'content' key")
                continue
                
            content = msg["content"]
            if not isinstance(content, list):
                print(f"    ERROR: Content is not a list: {type(content)}")
                continue
                
            for k, element in enumerate(content):
                if not isinstance(element, dict):
                    print(f"    ERROR: Content element {k} is not a dict: {type(element)}")
                    continue
                    
                if "type" not in element:
                    print(f"    ERROR: Content element {k} missing 'type' key")
                    continue
                    
                print(f"    Element {k}: {element['type']} - {list(element.keys())}")
    
    print(f"\nValidation complete for {min(max_samples, len(data))} samples.")


def simple_train_eval_split(
    data: List[Dict[str, Any]], eval_ratio: float, seed: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    eval_size = max(1, int(len(indices) * eval_ratio))
    eval_idx = set(indices[:eval_size])
    train = [data[i] for i in range(len(data)) if i not in eval_idx]
    eval_ = [data[i] for i in range(len(data)) if i in eval_idx]
    return train, eval_


def build_collator(processor):
    """Create a collator that tokenizes chat messages and optionally loads images.

    It masks labels so only the assistant response tokens are optimized by
    locating the response marker from the model's chat template. For Gemma 3,
    the marker '<start_of_turn>model\n' is commonly used.
    """

    response_marker = "<start_of_turn>model\n"
    marker_ids = processor.tokenizer(response_marker, add_special_tokens=False)[
        "input_ids"
    ]

    def process_vision_info(messages: list[dict]) -> list[Image.Image]:
        """Process vision information from messages"""
        image_inputs = []
        for msg in messages:
            content = msg.get("content", [])
            if not isinstance(content, list):
                content = [content]

            for element in content:
                if isinstance(element, dict) and element.get("type") == "image":
                    if "path" in element:
                        try:
                            img = Image.open(element["path"]).convert("RGB")
                            image_inputs.append(img)
                        except Exception as e:
                            print(f"Failed to open {element['path']}: {e}")
                    elif "image" in element:
                        img = element["image"]
                        if isinstance(img, Image.Image):
                            image_inputs.append(img.convert("RGB"))
        return image_inputs

    def collate(batch: List[Dict[str, Any]]):
        # Handle serialized messages
        processed_batch = []
        for ex in batch:
            processed_ex = ex.copy()
            if isinstance(ex.get("messages"), str):
                import json
                try:
                    processed_ex["messages"] = json.loads(ex["messages"])
                except json.JSONDecodeError:
                    # Fallback: treat as text-only
                    processed_ex["messages"] = [{"role": "user", "content": ex["messages"]}]
            processed_batch.append(processed_ex)
        
        # Render chat template strings
        texts: List[str] = [
            processor.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            ).strip()
            for ex in processed_batch
        ]

        # Collect images if available
        images_batch: List[List[Image.Image]] = []
        has_any_image = False
        for ex in processed_batch:
            imgs: List[Image.Image] = []
            
            # Check for image_path (legacy support)
            img_path = ex.get("image_path")
            if isinstance(img_path, str) and img_path:
                try:
                    with Image.open(img_path) as im:
                        imgs.append(im.convert("RGB"))
                        has_any_image = True
                except Exception:
                    # Fallback to text-only if image fails to open
                    pass
            
            # Check for images in messages (new format)
            if not imgs:
                imgs = process_vision_info(ex["messages"])
                if imgs:
                    has_any_image = True
            
            images_batch.append(imgs)

        if has_any_image:
            batch_tensors = processor(
                text=texts, images=images_batch, return_tensors="pt", padding=True
            )
        else:
            batch_tensors = processor(text=texts, return_tensors="pt", padding=True)

        # Label masking: ignore everything before the model response
        labels = batch_tensors["input_ids"].clone()
        marker_ids_tensor = torch.tensor(marker_ids, device=labels.device)
        for i, ids in enumerate(batch_tensors["input_ids"]):
            # Find first occurrence of marker_ids
            if len(marker_ids) > 0 and ids.numel() >= len(marker_ids):
                windows = ids.unfold(0, len(marker_ids), 1)
                matches = windows.eq(marker_ids_tensor).all(dim=1).nonzero(as_tuple=False)
                if len(matches) > 0:
                    start_idx = int(matches[0].item()) + len(marker_ids)
                    labels[i, :start_idx] = -100
                else:
                    # If we can't find marker, mask everything except last few tokens
                    labels[i, :-2] = -100
            else:
                labels[i, :-2] = -100

        batch_tensors["labels"] = labels
        return batch_tensors

    return collate


def print_data_samples(dataset, processor, num_samples=2):
    """Print detailed information of data samples"""
    print("\n" + "="*80)
    print("DATA SAMPLE EXAMPLES")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n{'='*20} Sample {i+1} {'='*20}")
        
        # Handle serialized messages
        messages = sample.get('messages', [])
        if isinstance(messages, str):
            import json
            try:
                messages = json.loads(messages)
            except json.JSONDecodeError:
                print(f"  ERROR: Could not deserialize messages: {messages[:100]}...")
                continue
        
        print(f"\nMessages structure:")
        for j, msg in enumerate(messages):
            print(f"  Message {j+1} ({msg.get('role', 'unknown')}):")
            
            content = msg.get('content', [])
            if not isinstance(content, list):
                content = [content]
            
            for k, element in enumerate(content):
                if isinstance(element, dict):
                    if element.get('type') == 'text':
                        text = element['text']
                        print(f"    Text {k+1}: {text}")
                    elif element.get('type') == 'image':
                        if 'path' in element:
                            print(f"    Image {k+1}: {element['path']}")
                        else:
                            print(f"    Image {k+1}: PIL Image object")
                    else:
                        print(f"    Content {k+1}: {type(element)}")
                else:
                    print(f"    Content {k+1}: {element}")
        
        print(f"\nComplete chat template:")
        try:
            full_text = processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            ).strip()
            print(f"  Full text: {full_text}")
                
        except Exception as e:
            print(f"  Error applying chat template: {e}")
        
        print(f"\nTokenization info:")
        try:
            tokens = processor.tokenizer(
                full_text, 
                add_special_tokens=False, 
                return_tensors="pt"
            )
            print(f"  Input IDs shape: {tokens['input_ids'].shape}")
            print(f"  Number of tokens: {tokens['input_ids'].shape[1]}")
            
            if tokens['input_ids'].shape[1] > 0:
                first_tokens = processor.tokenizer.convert_ids_to_tokens(
                    tokens['input_ids'][0]
                )
                print(f"  First 20 tokens: {first_tokens}")
                
        except Exception as e:
            print(f"  Error tokenizing: {e}")
        
        print(f"\n{'='*50}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--json_path", required=True, help="Path to JSON list dataset")
    p.add_argument("--model_name", default="google/gemma-3-4b-it")
    p.add_argument(
        "--output_dir", default="./runs_sft_gemma3_from_json", help="Output directory"
    )
    p.add_argument("--cache_dir", default="./hf_cache")

    # Split & limits
    p.add_argument("--eval_ratio", type=float, default=0.05)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)



    # Training hyperparams
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--num_epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=16)

    # LoRA
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)

    # Quantization
    p.add_argument("--use_4bit", action="store_true")

    # Eval config
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--eval_strategy", default="steps")
    
    # Early stopping
    p.add_argument("--early_stopping_patience", type=int, default=8,
                   help="Number of evaluations with no improvement after which training will be stopped")
    p.add_argument("--early_stopping_threshold", type=float, default=1e-4,
                   help="Denotes how much the specified metric must improve to satisfy early stopping conditions")

    # Misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--validate_data", action="store_true",
                   help="Validate data structure after conversion")

    return p.parse_args()


def main():
    args = get_args()

    # Accelerator for rank-aware printing
    accelerator = Accelerator()
    rank, is_main = accelerator.local_process_index, accelerator.is_main_process

    # Seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    data = load_json_dataset(args.json_path)
    
    # Always convert to new format
    if is_main:
        print("Converting legacy format to new message-embedded format...")
    data = convert_legacy_to_new_format(data)
    if is_main:
        print(f"Converted {len(data)} samples to new format")
        if args.validate_data:
            validate_converted_data(data, max_samples=3)
    
    train_rows, eval_rows = simple_train_eval_split(data, args.eval_ratio, args.seed)

    if args.max_train_samples:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_eval_samples:
        eval_rows = eval_rows[: args.max_eval_samples]

    # Build HF datasets with error handling
    try:
        train_ds = Dataset.from_list(train_rows)
        eval_ds = Dataset.from_list(eval_rows)
    except Exception as e:
        if is_main:
            print(f"Error creating datasets: {e}")
            print("Attempting to create datasets with string serialization...")
        
        # Fallback: serialize complex objects to strings
        def serialize_messages(sample):
            """Serialize messages to avoid Arrow type conflicts"""
            if "messages" in sample:
                import json
                sample["messages"] = json.dumps(sample["messages"])
            return sample
        
        train_rows_serialized = [serialize_messages(row.copy()) for row in train_rows]
        eval_rows_serialized = [serialize_messages(row.copy()) for row in eval_rows]
        
        train_ds = Dataset.from_list(train_rows_serialized)
        eval_ds = Dataset.from_list(eval_rows_serialized)
        
        if is_main:
            print("Created datasets with serialized messages")

    if is_main:
        print(f"Training dataset: {len(train_ds)} samples")
        print(f"Evaluation dataset: {len(eval_ds)} samples")

    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_name, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if getattr(processor, "tokenizer", None) and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if getattr(processor, "tokenizer", None):
        processor.tokenizer.padding_side = "right"

    # Add 500 control tokens and record how many were new
    toks = [f"[control_{i}]" for i in range(1, 501)]
    n_new = processor.tokenizer.add_special_tokens({"additional_special_tokens": toks})
    if is_main:
        print(f"Added {len(toks)} control tokens (1-500); newly added: {n_new}")    
        print_data_samples(train_ds, processor, num_samples=2)
        print_data_samples(eval_ds, processor, num_samples=1)

    # Model
    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    
    # Only use device_map for single GPU or XPU
    if not torch.distributed.is_initialized():
        model_kwargs["device_map"] = {"": f"xpu:{rank}"} if is_xpu_available() else "auto"
    if args.use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    Model = (
        Gemma3ForConditionalGeneration
        if "gemma-3" in args.model_name.lower()
        else AutoModelForCausalLM
    )
    model = Model.from_pretrained(args.model_name, **model_kwargs)
    model.config.use_cache = False

    # Resize embeddings to accommodate new tokens and initialize them
    old_vocab_size = model.get_input_embeddings().weight.size(0)
    if len(processor.tokenizer) > old_vocab_size:
        if is_main:
            print(f"Resizing token embeddings: {old_vocab_size} -> {len(processor.tokenizer)}")
        
        # Resize embeddings on CPU first to avoid GPU memory issues
        model = model.cpu()
        model.resize_token_embeddings(len(processor.tokenizer))
        
        # Initialize new embeddings with mean of existing ones
        input_embeddings = model.get_input_embeddings()
        new_token_start = old_vocab_size
        
        with torch.no_grad():
            mean_vec = input_embeddings.weight.data[:old_vocab_size].mean(dim=0)
            input_embeddings.weight.data[new_token_start:] = mean_vec
        
        if is_main:
            print(f"New embedding rows initialized from mean vector; example ID of [control_10]: {processor.tokenizer('[control_10]', add_special_tokens=False)['input_ids']}")
        
        # Let Accelerate/DeepSpeed handle GPU distribution
        # The model will be automatically moved to the correct device by the trainer

    # Tie weights between embed_tokens and lm_head
    model.tie_weights()

    # LoRA config (no modules_to_save)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
        auto_mapping=True,
    )

    # Collator
    collator = build_collator(processor)

    # TrainingArguments
    targs = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=5,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_steps=args.eval_steps,
        save_strategy="steps",
        save_total_limit=1,
        report_to="tensorboard",
        remove_unused_columns=False,
        group_by_length=False,
        run_name=f"gemma3-from-json-{datetime.now():%Y%m%d-%H%M%S}",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Callback: show decoded prediction at eval
    class ShowPredsCallback(TrainerCallback):
        def __init__(self, processor, eval_dataset, num_examples: int = 1):
            self.processor = processor
            self.eval_dataset = eval_dataset
            self.num_examples = num_examples
            self.trainer = None

        def set_trainer(self, trainer):
            self.trainer = trainer

        def on_evaluate(self, args, state, control, **kwargs):
            if self.trainer is None or not is_main:
                return control

            model = self.trainer.model
            tokenizer = self.processor.tokenizer
            device = model.device
            END_ID = tokenizer.convert_tokens_to_ids("<end_of_turn>")

            for i in range(min(self.num_examples, len(self.eval_dataset))):
                chat = self.eval_dataset[i]["messages"][:-1]
                try:
                    prompt = self.processor.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    prompt_len = inputs["input_ids"].size(1)

                    with torch.no_grad():
                        gen_ids = model.generate(
                            **inputs,
                            max_new_tokens=1,
                            do_sample=False,
                            eos_token_id=END_ID,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    new_token_ids = gen_ids[0][prompt_len:]
                    pred_text = tokenizer.decode(
                        new_token_ids, skip_special_tokens=False
                    )

                    print("\n=== SAMPLE PREDICTION ===")
                    print(
                        "Gold :",
                        self.eval_dataset[i]["messages"][-1]["content"],
                    )
                    print("Pred :", pred_text)
                    print("=========================\n")
                except Exception as e:
                    print(f"Prediction preview failed for sample {i}: {e}")

            return control

    # cb = ShowPredsCallback(processor, eval_ds, num_examples=2)
    early_stop_cb = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience, 
        early_stopping_threshold=args.early_stopping_threshold
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=targs,
        peft_config=lora,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=processor,
        data_collator=collator,
        callbacks=[early_stop_cb],
    )

    # Unfreeze embed_tokens parameters
    embed_tokens = trainer.model.get_input_embeddings()
    # Enable gradient computation for embed_tokens weights to train the newly added control token embeddings
    embed_tokens.weight.requires_grad_(True)
    if is_main:
        print("Unfrozen embed_tokens parameters")

    # Diagnostics: trainable flags and tie status
    if is_main:
        print("\n" + "=" * 60)
        print("Checking trainable status of embedding layers")
        print("=" * 60)

        embed_tokens = trainer.model.get_input_embeddings()
        lm_head = trainer.model.get_output_embeddings()

        print(
            f"Embedding layer (embed_tokens) trainable: {embed_tokens.weight.requires_grad}"
        )
        print(
            f"Output layer (lm_head) trainable: {lm_head.weight.requires_grad}"
        )

        embed_weight = embed_tokens.weight
        lm_head_weight = lm_head.weight
        weights_tied = torch.equal(embed_weight, lm_head_weight)
        print(f"Weights tied between embed_tokens and lm_head: {weights_tied}")

        if weights_tied:
            print("✓ Weight tying is working correctly")
        else:
            print("✗ Weight tying is NOT working - weights are different!")

        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        print(f"Total number of parameters: {total_params:,}")
        print(f"Number of trainable parameters: {trainable_params:,}")
        print(
            f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%"
        )
        print("=" * 60 + "\n")

        with open("params_info.txt", "w") as f:
            for name, p in model.named_parameters():
                f.write(f"{name} {p.requires_grad} {p.shape}\n")

    # Ensure callback can access the trainer
    # cb.set_trainer(trainer)

    # Train
    trainer.train()

    # Merge LoRA and save final model
    if is_main:
        print("\n" + "=" * 60)
        print("Merging LoRA weights and saving final model")
        print("=" * 60)

    trainer.model = trainer.model.merge_and_unload()

    # Check final tie status
    if is_main:
        embed_tokens = trainer.model.get_input_embeddings()
        lm_head = trainer.model.get_output_embeddings()
        final_weights_tied = torch.equal(embed_tokens.weight, lm_head.weight)
        print(f"Final weights tied after merging: {final_weights_tied}")
        if final_weights_tied:
            print("✓ Weight tying maintained after LoRA merging")
        else:
            print("✗ Weight tying lost after LoRA merging!")

        ckpt = os.path.join(args.output_dir, "final_checkpoint")
        trainer.model.save_pretrained(ckpt)
        processor.save_pretrained(ckpt)
        print(f"Saved merged model to {ckpt}")
        print("=" * 60)


if __name__ == "__main__":
    main() 