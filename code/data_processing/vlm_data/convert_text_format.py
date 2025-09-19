#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen → Simple message converter
• Keeps the original system prompt
• Removes any additional system blocks
• Drops a leading assistant greeting (if present) so that the first
  post-system turn is always a user message.
• Converts any numeric topic label (1-9 → [control_1] … 10 → [control_10] …)
  using four carefully scoped regex patterns to avoid touching ordinary numbers.
"""

import json
import os
import re
from typing import List, Union
from tqdm import tqdm
# Removed transformers import since we're not using Mistral tokenizer anymore

# --------------------------------------------------------------------------- #
# 0.  Model paths – edit to match your setup
# --------------------------------------------------------------------------- #
# Removed MISTRAL_PATH and MISTRAL_tok since we're not applying Mistral format

# --------------------------------------------------------------------------- #
# 1.  Helper functions
# --------------------------------------------------------------------------- #
def extract_messages_from_qwen(text: str):
    """Qwen raw string → list[dict(role, content)]."""
    chunks = [c for c in text.split("<|im_start|>") if c.strip()]
    messages = []
    for chunk in chunks:
        idx = chunk.find("\n")
        if idx == -1:
            continue
        role    = chunk[:idx].strip()
        content = chunk[idx:].replace("<|im_end|>", "").strip()
        messages.append({"role": role, "content": content})
    return messages


def merge_consecutive(msgs):
    """Merge consecutive identical-role messages."""
    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n" + m["content"].lstrip()
        else:
            merged.append(m.copy())
    return merged


def strip_extra_system_messages(msgs):
    """Remove every system entry unless it is the very first element."""
    return [m for i, m in enumerate(msgs) if not (m["role"] == "system" and i != 0)]


def drop_leading_assistant_greeting(msgs):
    """
    If the first non-system turn is 'assistant', drop it so the first user
    request comes directly after the system prompt.
    """
    i = 0
    if msgs and msgs[0]["role"] == "system":
        i = 1
    if len(msgs) > i and msgs[i]["role"] == "assistant":
        msgs.pop(i)
    return msgs


def keep_last_assistant(messages):
    """
    Collapse any consecutive assistant messages at the end into a single entry.
    """
    if not messages:
        return messages

    idx = len(messages) - 1
    if messages[idx]["role"] != "assistant":
        return messages

    trim_point = idx - 1
    while trim_point >= 0 and messages[trim_point]["role"] == "assistant":
        trim_point -= 1

    return messages[: trim_point + 1] + [messages[idx]]


# --------------------------------------------------------------------------- #
# 2.  Regex patterns & CONTROL mapping
# --------------------------------------------------------------------------- #
numbered_pattern          = re.compile(r'\n([1-9]\d?)(?=\s)')
control_pattern           = re.compile(r'\[control_(\d+)\]')
label_after_think_pattern = re.compile(r'</think>\s*\n?\s*(?:\[control_(\d+)\]|(\d+))')
stray_end_number_pattern  = re.compile(r'(\n|\A)\s*([1-9]\d?)\s*(?=(</s>)?$)')


def make_control(n: int) -> str:
    """Convert integer n to [control_n]."""
    return f"[control_{n}]"


# --------------------------------------------------------------------------- #
# 3.  Single-prompt conversion
# --------------------------------------------------------------------------- #
def convert_single_prompt(qwen_prompt: str) -> Union[List, None]:
    msgs = extract_messages_from_qwen(qwen_prompt)
    msgs = merge_consecutive(msgs)

    # Skip multi-turn conversations - only keep single-turn conversations
    user_count = sum(1 for msg in msgs if msg["role"] == "user")
    assistant_count = sum(1 for msg in msgs if msg["role"] == "assistant")
    
    # Skip if there are multiple user or assistant messages (multi-turn)
    if user_count > 1 or assistant_count > 1:
        return None

    is_topic = (
        msgs
        and msgs[0]["role"] == "system"
        and msgs[0]["content"].lstrip().lower().startswith(
            "you are a topic classification expert"
        )
    )

    if is_topic:
        # Find the last user message and add the marker
        for i in reversed(range(len(msgs))):
            if msgs[i]["role"] == "user":
                msgs[i]["content"] = (
                    "### USER CONVERSATION HERE ###\n" + msgs[i]["content"].lstrip()
                )
                break

    # Keep system message separate - don't merge into user content
    # The system message will remain as a separate entry with role="system"

    msgs = strip_extra_system_messages(msgs)
    msgs = drop_leading_assistant_greeting(msgs)
    msgs = keep_last_assistant(msgs)

    if is_topic:
        for i in reversed(range(len(msgs))):
            if msgs[i]["role"] == "user":
                msgs[i]["content"] = msgs[i]["content"].rstrip() + (
                    "\n\nBased on the above conversation, respond with the relevant "
                    "topic ID:\n"
                )
                break

    # Apply regex transformations to each message content
    for msg in msgs:
        content = msg["content"]
        
        # Remove artifacts
        content = re.sub(
            r'RULE:\s*For topic IDs >=10 use \[control_\d+\](?:,\s*\[control_\d+\])*,\s*etc\.\s*',
            '',
            content,
        )
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)

        # --- numeric label → [control_x] ---
        content = numbered_pattern.sub(
            lambda m: f"\n{make_control(int(m.group(1)))}",
            content,
        )
        content = control_pattern.sub(
            lambda m: make_control(int(m.group(1))),
            content,
        )
        content = label_after_think_pattern.sub(
            lambda m: make_control(int(m.group(1) or m.group(2))),
            content,
        )
        content = stray_end_number_pattern.sub(
            lambda m: f"{m.group(1)}{make_control(int(m.group(2)))}",
            content,
        )
        
        msg["content"] = content.strip()

    return msgs


# --------------------------------------------------------------------------- #
# 4.  Batch processing
# --------------------------------------------------------------------------- #
def main():
    src_json = "full_sft_merged.json"
    out_json = "text.json"

    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    converted = []
    skipped = 0

    for item in tqdm(data, desc="Converting"):
        new_item = item.copy()
        new_messages = convert_single_prompt(item["prompt"])
        if new_messages is None:
            skipped += 1
            continue
        new_item["messages"] = new_messages
        # Remove the old "prompt" field since we're now using "messages"
        if "prompt" in new_item:
            del new_item["prompt"]
        converted.append(new_item)

    # Only create directory if the output path has a directory component
    output_dir = os.path.dirname(out_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(converted, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {len(converted)} conversations saved to → {out_json}")
    print(f"✗ {skipped} conversations skipped due to errors")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
