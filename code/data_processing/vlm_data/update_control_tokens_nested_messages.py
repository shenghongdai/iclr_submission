#!/usr/bin/env python3
"""
Update control tokens in JSON files with nested messages structure
Randomly assign numbers 1-500 to each control token and ensure answer consistency
Then combine both files together
"""

import json
import random
import re
from pathlib import Path

def extract_control_tokens(text):
    """Extract all control tokens from text using regex."""
    pattern = r'\[control_(\d+)\]'
    matches = re.findall(pattern, text)
    return [int(match) for match in matches]

def replace_control_tokens(text, token_mapping):
    """Replace control tokens in text with new random numbers."""
    def replace_match(match):
        old_token = int(match.group(1))
        new_token = token_mapping.get(old_token, old_token)
        return f'[control_{new_token}]'
    
    return re.sub(r'\[control_(\d+)\]', replace_match, text)

def remove_control_tokens(text):
    """Remove all control tokens from text completely."""
    return re.sub(r'\[control_\d+\]', '', text)

def update_control_tokens_in_messages_entry(entry):
    """Update control tokens in an entry that has a messages array."""
    if 'messages' not in entry:
        return entry, False
    
    # Extract all control tokens from all messages in the entry
    all_tokens = set()
    
    for message in entry['messages']:
        if 'content' in message:
            tokens = extract_control_tokens(message['content'])
            all_tokens.update(tokens)
    
    if not all_tokens:
        return entry, False
    
    # Create random mapping for tokens 1-500
    available_numbers = list(range(1, 501))
    token_mapping = {}
    
    for old_token in sorted(all_tokens):
        if available_numbers:
            new_token = random.choice(available_numbers)
            token_mapping[old_token] = new_token
            available_numbers.remove(new_token)
    
    # Apply the mapping to all messages in the entry
    for message in entry['messages']:
        if 'content' in message:
            message['content'] = replace_control_tokens(message['content'], token_mapping)
    
    # Apply the same mapping to correct_answer if it exists
    if 'correct_answer' in entry:
        entry['correct_answer'] = replace_control_tokens(entry['correct_answer'], token_mapping)
    
    return entry, True

def update_control_tokens_in_image_entry(entry):
    """Update control tokens in image classification entry."""
    # Extract all control tokens from the entry
    all_tokens = set()
    
    # Check instruction field
    if 'instruction' in entry:
        tokens = extract_control_tokens(entry['instruction'])
        all_tokens.update(tokens)
    
    # Check content field
    if 'content' in entry:
        tokens = extract_control_tokens(entry['content'])
        all_tokens.update(tokens)
    
    # Check correct_answer field and include in mapping
    if 'correct_answer' in entry:
        tokens = extract_control_tokens(entry['correct_answer'])
        all_tokens.update(tokens)
    
    if not all_tokens:
        return entry, False
    
    # Create random mapping for tokens 1-500
    available_numbers = list(range(1, 501))
    token_mapping = {}
    
    for old_token in sorted(all_tokens):
        if available_numbers:
            new_token = random.choice(available_numbers)
            token_mapping[old_token] = new_token
            available_numbers.remove(new_token)
    
    # Apply the mapping to all fields including correct_answer
    if 'instruction' in entry:
        entry['instruction'] = replace_control_tokens(entry['instruction'], token_mapping)
    
    if 'content' in entry:
        entry['content'] = replace_control_tokens(entry['content'], token_mapping)
    
    if 'correct_answer' in entry:
        entry['correct_answer'] = replace_control_tokens(entry['correct_answer'], token_mapping)
    
    return entry, True

def process_json_file_with_messages(input_file, output_file):
    """Process a JSON file with messages structure to update control tokens."""
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} entries...")
    
    updated_count = 0
    for i, entry in enumerate(data):
        # Check if this entry has messages structure
        if 'messages' in entry:
            updated_entry, was_updated = update_control_tokens_in_messages_entry(entry)
        else:
            # Handle as image classification entry
            updated_entry, was_updated = update_control_tokens_in_image_entry(entry)
        
        if was_updated:
            updated_count += 1
        
        # Progress indicator
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(data)} entries...")
    
    print(f"Updated {updated_count} entries with new control tokens")
    
    # Save updated file
    print(f"Saving updated file to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return data

def combine_json_files(file1, file2, output_file):
    """Combine two JSON files into one."""
    print(f"Combining {file1} and {file2}...")
    
    # Read both files
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)
    
    # Combine the data
    combined_data = data1 + data2
    
    print(f"Combined {len(data1)} entries from {file1}")
    print(f"Combined {len(data2)} entries from {file2}")
    print(f"Total entries: {len(combined_data)}")
    
    # Save combined file
    print(f"Saving combined file to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=2, ensure_ascii=False)
    
    return combined_data

def main():
    print("="*60)
    print("UPDATING CONTROL TOKENS IN NESTED MESSAGES STRUCTURE")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # File paths
    image_file = 'image_classification_updated.json'
    text_file = 'text.json'
    updated_image_file = 'image_classification_control_updated_nested.json'
    updated_text_file = 'text_control_updated_nested.json'
    combined_file = 'combined_control_updated_nested.json'
    
    try:
        # Check if files exist
        if not Path(image_file).exists():
            print(f"❌ Error: {image_file} not found")
            return 1
        
        if not Path(text_file).exists():
            print(f"❌ Error: {text_file} not found")
            return 1
        
        # Process image classification file
        print("\n1. Processing image classification file...")
        image_data = process_json_file_with_messages(image_file, updated_image_file)
        
        # Process text file
        print("\n2. Processing text file...")
        text_data = process_json_file_with_messages(text_file, updated_text_file)
        
        # Combine files
        print("\n3. Combining files...")
        combined_data = combine_json_files(updated_image_file, updated_text_file, combined_file)
        
        # Show some examples
        print("\n📋 Example updates:")
        print("\nImage classification examples:")
        for i, entry in enumerate(image_data[:3]):
            if 'instruction' in entry:
                instruction = entry['instruction'][:100] + "..." if len(entry['instruction']) > 100 else entry['instruction']
                print(f"  Entry {i+1}: {instruction}")
            elif 'messages' in entry:
                first_msg = entry['messages'][0]['content'][:100] + "..." if len(entry['messages'][0]['content']) > 100 else entry['messages'][0]['content']
                print(f"  Entry {i+1} (messages): {first_msg}")
        
        print("\nText examples:")
        for i, entry in enumerate(text_data[:3]):
            if 'messages' in entry:
                first_msg = entry['messages'][0]['content'][:100] + "..." if len(entry['messages'][0]['content']) > 100 else entry['messages'][0]['content']
                print(f"  Entry {i+1}: {first_msg}")
        
        print("\n" + "="*60)
        print("UPDATE COMPLETED")
        print("="*60)
        print(f"✅ Updated image file: {updated_image_file}")
        print(f"✅ Updated text file: {updated_text_file}")
        print(f"✅ Combined file: {combined_file}")
        print(f"📊 Total entries: {len(combined_data)}")
        print(f"🎲 Control tokens randomly assigned from 1-500")
        print(f" Answer consistency maintained")
        print(f"💬 Nested messages structure properly handled")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main() 