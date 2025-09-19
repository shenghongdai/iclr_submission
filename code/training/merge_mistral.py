#!/usr/bin/env python3
"""
Merge Mistral-Small-3.1-24B base with LoRA adapter.
No need to untie lm_head – Mistral is naturally untied.

⚑  Requires transformers ≥ 4.41
"""

import os, torch
from transformers import Mistral3ForConditionalGeneration, AutoProcessor
from peft import PeftModel

# ─────────────── Config ──────────────────
BASE_MODEL = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
PEFT_DIR   = "./runs_sft_mistral3_24b_full_fast/final_checkpoint"
OUT_DIR    = "./merged_mistral24b_full_fast"
CACHE_DIR  = "./hf_cache"
DEVICE     = "auto"  # or "cuda:0"
TEST_TEXT  = "Thank you so much for your help!"
# ─────────────────────────────────────────

def test_model(model, proc, text=TEST_TEXT):
    """Generate one token – the predicted control ID."""
    topics = "\n".join(
        f"[control_{i+1}] {l}\n#####"
        for i, l in enumerate([
            'Acknowledge','Advise','Agree','Apologise','Arrange','Ask for help',
            'Asking for opinions','Care','Comfort','Complain','Confirm','Criticize',
            'Doubt','Emphasize','Explain','Flaunt','Greet','Inform','Introduce',
            'Invite','Joke','Leave','Oppose','Plan','Praise','Prevent','Refuse',
            'Taunt','Thank','Warn'])
    )
    system = (
        "You are a topic classification expert. Before making a decision, "
        "carefully follow all the topic-specific instructions/descriptions.\n"
        "Topics:\n" + topics
    )
    msg = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": [{"type": "text", "text":
            f"### USER CONVERSATION HERE ###\n{text}\n\n"
            "Based on the above conversation, respond with the relevant topic ID:\n"}]},
    ]
    prompt = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    toks   = proc(text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **toks, max_new_tokens=1, do_sample=False,
            eos_token_id=model.config.eos_token_id or proc.tokenizer.eos_token_id,
            pad_token_id=proc.tokenizer.eos_token_id,
        )
    return proc.tokenizer.decode(out[0][toks["input_ids"].size(1):]).strip()

# ───── Load base & merge PEFT adapter ─────
print("▶ Loading base model …")
base = Mistral3ForConditionalGeneration.from_pretrained(
    BASE_MODEL, device_map=DEVICE, torch_dtype=torch.bfloat16,
    trust_remote_code=True, cache_dir=CACHE_DIR
)

print("▶ Loading processor …")
proc = AutoProcessor.from_pretrained(PEFT_DIR, trust_remote_code=True, cache_dir=CACHE_DIR)

if len(proc.tokenizer) > base.get_input_embeddings().num_embeddings:
    base.resize_token_embeddings(len(proc.tokenizer))

print("▶ Loading LoRA adapter …")
peft = PeftModel.from_pretrained(base, PEFT_DIR)

print("▶ Merging adapter into base …")
merged = peft.merge_and_unload()

# ───── Save merged model ─────
os.makedirs(OUT_DIR, exist_ok=True)
print(f"▶ Saving to {OUT_DIR} …")
merged.save_pretrained(OUT_DIR, safe_serialization=True)
proc.save_pretrained(OUT_DIR)

print("▶ Quick test (in-memory):", test_model(merged, proc))

# ───── Simulate reload ─────
print("▶ Reloading merged model …")
model = Mistral3ForConditionalGeneration.from_pretrained(
    OUT_DIR, device_map=DEVICE, torch_dtype=torch.bfloat16,
    trust_remote_code=True, cache_dir=CACHE_DIR
)
proc = AutoProcessor.from_pretrained(OUT_DIR, trust_remote_code=True, cache_dir=CACHE_DIR)

print("▶ Smoke-test after reload:", test_model(model, proc))
