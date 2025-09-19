#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create image.json with classification problem format similar to text.json
Each conversation has multiple choice options with [control_x] labels
Combines data from:
- MIntRec: Video segments with intent classification
- AOKVQA: Visual question answering with COCO images
- Visual7W: Visual question answering with custom images
"""

import json
import os
import random
from typing import List, Dict, Any
from tqdm import tqdm
import pandas as pd

# Set random seed for reproducibility
random.seed(42)

# --------------------------------------------------------------------------- #
# 1. Helper functions
# --------------------------------------------------------------------------- #

def create_mintrec_classification_conversations(mintrec_data_path: str, max_samples: int = None) -> List[Dict]:
    """Create classification conversations from MIntRec dataset."""
    conversations = []
    
    # Read all MIntRec TSV files (train, dev, test)
    train_file = os.path.join(mintrec_data_path, 'train.tsv')
    dev_file = os.path.join(mintrec_data_path, 'dev.tsv')
    test_file = os.path.join(mintrec_data_path, 'test.tsv')
    
    all_data = []
    for file_path, split_name in [(train_file, 'train'), (dev_file, 'dev'), (test_file, 'test')]:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep='\t')
            data = df.to_dict('records')
            # Add split information to each record
            for item in data:
                item['split'] = split_name
            all_data.extend(data)
    
    # Use all MIntRec data (train + dev + test)
    print(f"Using all {len(all_data)} MIntRec samples (train + dev + test)")
    
    # Define all 20 MIntRec intent labels in order
    all_intent_labels = [
        # Express emotions and attitudes
        'Complain', 'Praise', 'Apologise', 'Thank', 'Criticize', 
        'Care', 'Agree', 'Taunt', 'Flaunt', 'Oppose', 'Joke',
        # Achieve goals  
        'Inform', 'Advise', 'Arrange', 'Introduce', 'Comfort', 
        'Leave', 'Prevent', 'Greet', 'Ask for help'
    ]
    
    for item in all_data:
        # Create video path
        video_path = f"MIntRec/data/MIntRec/{item['season']}/{item['episode']}/{item['clip']}.mp4"
        
        # Per-sample shuffled intent list
        labels = all_intent_labels.copy()
        random.shuffle(labels)

        # Create multiple choice options with [control_x] format using the shuffled intents
        options = [f"[control_{i+1}] {label}" for i, label in enumerate(labels)]

        # Find the correct answer index in the shuffled list
        correct_idx = labels.index(item['label'])
        correct_answer = options[correct_idx]
        
        # Create topics text for system role with ##### separators
        topics_text = "\n#####\n".join(options)
        
        # Create conversation with system role containing topics
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a multimodal intent classification expert. Before making a decision, carefully follow all the topic-specific instructions and descriptions.\n\nTopics:\n{topics_text}"
                },
                {
                    "role": "user",
                    "content": f"### USER CONVERSATION HERE ###\n\n'{item['text']}'"
                },
                {
                    "role": "assistant", 
                    "content": f"[control_{correct_idx + 1}]"
                }
            ],
            "image_path": video_path,
            "dataset": "MIntRec",
            "intent_label": item['label'],
            "split": item['split'],
            "correct_answer": correct_answer,
            "correct_idx": correct_idx
        }
        conversations.append(conversation)
    
    return conversations

def create_aokvqa_classification_conversations(aokvqa_data_path: str, coco_path: str, max_samples: int = 5000) -> List[Dict]:
    """Create classification conversations from AOKVQA dataset."""
    conversations = []
    
    # Load only AOKVQA train data
    train_file = os.path.join(aokvqa_data_path, 'aokvqa_v1p0_train.json')
    
    all_data = []
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    # Sample part of AOKVQA training data
    if len(all_data) > max_samples:
        all_data = random.sample(all_data, max_samples)
    print(f"Using {len(all_data)} out of {len(all_data)} AOKVQA training samples")
    
    for item in all_data:
        # Create image path
        image_path = f"aokvqa/datasets/coco/train2017/{item['image_id']:012d}.jpg"
        
        # Create multiple choice options with [control_x] format
        options = []
        correct_idx = item['correct_choice_idx']
        
        for i, choice in enumerate(item['choices']):
            options.append(f"[control_{i+1}] {choice}")
        
        # Create topics text for system role with ##### separators
        topics_text = "\n#####\n".join(options)
        
        # Create conversation with system role containing topics
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a visual question answering expert. Before making a decision, carefully follow all the topic-specific instructions and descriptions.\n\nTopics:\n{topics_text}"
                },
                {
                    "role": "user",
                    "content": f"### USER CONVERSATION HERE ###\n\n'{item['question']}'"
                },
                {
                    "role": "assistant",
                    "content": f"[control_{correct_idx + 1}]"
                }
            ],
            "image_path": image_path,
            "dataset": "AOKVQA",
            "question": item['question'],
            "correct_answer": item['choices'][correct_idx],
            "correct_idx": correct_idx
        }
        conversations.append(conversation)
    
    return conversations

def create_visual7w_classification_conversations(visual7w_data_path: str, images_path: str, max_samples: int = 5000) -> List[Dict]:
    """Create classification conversations from Visual7W dataset."""
    conversations = []
    
    # Load Visual7W data
    dataset_file = os.path.join(visual7w_data_path, 'dataset.json')
    if not os.path.exists(dataset_file):
        return conversations
    
    with open(dataset_file, 'r') as f:
        data = json.load(f)
    
    # Collect only train QA pairs
    all_qa_pairs = []
    for image in data['images']:
        # Only use train split images
        if image.get('split') == 'train':
            for qa_pair in image['qa_pairs']:
                qa_pair['image_filename'] = image['filename']
                all_qa_pairs.append(qa_pair)
    
    # Sample part of Visual7W training data
    if len(all_qa_pairs) > max_samples:
        all_qa_pairs = random.sample(all_qa_pairs, max_samples)
    print(f"Using {len(all_qa_pairs)} out of {len(all_qa_pairs)} Visual7W training samples")
    
    for qa_pair in all_qa_pairs:
        # Create image path
        image_path = f"visual7w-toolkit/images/{qa_pair['image_filename']}"
        
        # Gather choices as plain texts (no control tags yet)
        correct_answer = qa_pair['answer']
        choices = [correct_answer] + list(qa_pair['multiple_choices'])
        
        # Shuffle choices and find correct index
        random.shuffle(choices)
        correct_idx = choices.index(correct_answer)
        
        # Now assign control tokens sequentially by position
        labeled_options = [f"[control_{i+1}] {choice}" for i, choice in enumerate(choices)]
        topics_text = "\n#####\n".join(labeled_options)
        
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a visual question answering expert. Before making a decision, carefully follow all the topic-specific instructions and descriptions.\n\nTopics:\n{topics_text}"
                },
                {
                    "role": "user",
                    "content": f"### USER CONVERSATION HERE ###\n\n'{qa_pair['question']}'"
                },
                {
                    "role": "assistant",
                    "content": f"[control_{correct_idx + 1}]"
                }
            ],
            "image_path": image_path,
            "dataset": "Visual7W",
            "question": qa_pair['question'],
            "answer": qa_pair['answer'],
            "question_type": qa_pair['type'],
            "correct_answer": f"[control_{correct_idx + 1}] {correct_answer}",
            "correct_idx": correct_idx
        }
        conversations.append(conversation)
    
    return conversations

# --------------------------------------------------------------------------- #
# 2. Main function
# --------------------------------------------------------------------------- #

def main():
    """Create image.json with classification format."""
    
    # Configuration
    output_file = "image_classification.json"
    # MIntRec: use all data (train + dev + test)
    # AOKVQA: use 5000 samples from training data
    # Visual7W: use 5000 samples from training data
    
    print("Creating image_classification.json with multimodal classification data...")
    
    all_conversations = []
    
    # 1. Add MIntRec data (all data: train + dev + test)
    print("Processing MIntRec data...")
    mintrec_conversations = create_mintrec_classification_conversations(
        "MIntRec/data/MIntRec"
    )
    all_conversations.extend(mintrec_conversations)
    print(f"Added {len(mintrec_conversations)} MIntRec conversations")
    
    # 2. Add AOKVQA data (5000 samples from training)
    print("Processing AOKVQA data...")
    aokvqa_conversations = create_aokvqa_classification_conversations(
        "aokvqa/datasets/aokvqa",
        "aokvqa/datasets/coco",
        max_samples=5000
    )
    all_conversations.extend(aokvqa_conversations)
    print(f"Added {len(aokvqa_conversations)} AOKVQA conversations")
    
    # 3. Add Visual7W data (5000 samples from training)
    print("Processing Visual7W data...")
    visual7w_conversations = create_visual7w_classification_conversations(
        "visual7w-toolkit/datasets/visual7w-telling",
        "visual7w-toolkit/images",
        max_samples=5000
    )
    all_conversations.extend(visual7w_conversations)
    print(f"Added {len(visual7w_conversations)} Visual7W conversations")
    
    # Shuffle all conversations
    random.shuffle(all_conversations)
    
    # Save to file
    print(f"Saving {len(all_conversations)} total conversations to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, ensure_ascii=False, indent=2)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total conversations: {len(all_conversations)}")
    
    dataset_counts = {}
    for conv in all_conversations:
        dataset = conv['dataset']
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    for dataset, count in dataset_counts.items():
        print(f"{dataset}: {count} conversations")
    
    print(f"\nFile saved as: {output_file}")
    print(f"File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main() 