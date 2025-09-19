#!/usr/bin/env python3
"""
SST-2 Evaluation Script for Finetuned Models using vLLM (Custom Control Tokens)

This script evaluates a finetuned model on the Stanford Sentiment Treebank (SST-2) dataset using control tokens,
which contains movie reviews for binary sentiment analysis:
Positive (label=1) and Negative (label=0).
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer, AutoProcessor
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')
import time
import multiprocessing as mp
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# vLLM specific imports and configuration
# Set multiprocessing method to spawn for vLLM compatibility
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
mp.set_start_method("spawn", force=True)
from vllm import LLM, SamplingParams
from datasets import load_dataset

class SST2DataLoader:
    """
    Data loader for SST-2 dataset from Hugging Face datasets.
    
    This class handles loading and preprocessing the SST-2 dataset,
    which contains movie reviews for binary sentiment analysis.
    """
    
    def __init__(self, dataset_name: str = 'glue', subset: str = 'sst2'):
        """
        Initialize the SST-2 data loader.
        
        Args:
            dataset_name: Name of the dataset from Hugging Face datasets
            subset: Subset name (default: 'sst2')
        """
        self.dataset_name = dataset_name
        self.subset = subset
        
        # Define the 2 sentiment classes for binary classification
        # 0: Negative, 1: Positive
        self.class_labels = ['Negative', 'Positive']
        self.label_to_id = {label: idx for idx, label in enumerate(self.class_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Load the dataset from Hugging Face
        print(f"Loading {dataset_name}/{subset} dataset from Hugging Face...")
        self.dataset = load_dataset(dataset_name, subset)
        
        # Process the dataset
        self.train_data = self._process_split('train')
        self.validation_data = self._process_split('validation')
        self.test_data = self._process_split('test') if 'test' in self.dataset else []
        
        print(f"Loaded {len(self.train_data)} train, {len(self.validation_data)} validation, {len(self.test_data)} test samples from {dataset_name}/{subset}")
        print(f"Dataset structure: {self.dataset}")
    
    def _process_split(self, split: str) -> List[Dict]:
        """
        Process data from a specific split (train/validation/test).
        
        Args:
            split: Split name ('train', 'validation', or 'test')
            
        Returns:
            List of dictionaries containing sample data
        """
        data = []
        if split not in self.dataset:
            print(f"Warning: {split} split not found in dataset. Skipping.")
            return data
            
        split_data = self.dataset[split]
        
        print(f"Processing {split} split with {len(split_data)} samples...")
        
        for idx, sample in enumerate(split_data):
            try:
                # Extract data from Hugging Face dataset format
                sentence = sample.get('sentence', '')
                label = sample.get('label', 0)  # 0: Negative, 1: Positive
                idx_field = sample.get('idx', idx)  # Original index from dataset
                
                # Convert label to our format
                label_name = self.class_labels[label]  # 0 -> 'Negative', 1 -> 'Positive'
                
                sample_data = {
                    'index': idx,
                    'original_idx': idx_field,
                    'label_id': label,
                    'label': label_name,
                    'sentence': sentence,
                    'text': sentence,  # For compatibility with evaluator
                    'split': split
                }
                data.append(sample_data)
                
            except Exception as e:
                print(f"Warning: sample {idx} processing error: {e}")
                continue
        
        return data
    
    def get_sample(self, idx: int, split: str = 'validation') -> Optional[Dict]:
        """
        Get a specific sample by index.
        
        Args:
            idx: Sample index
            split: Split to get sample from ('train', 'validation', or 'test')
            
        Returns:
            Sample dictionary or None if index out of range
        """
        if split == 'train':
            data = self.train_data
        elif split == 'validation':
            data = self.validation_data
        elif split == 'test':
            data = self.test_data
        else:
            return None
            
        if 0 <= idx < len(data):
            return data[idx]
        return None


class SST2FlashTopicEvaluator:
    """
    Evaluator for models on SST-2 dataset using vLLM with control tokens.
    
    This class provides comprehensive evaluation capabilities for models
    on the SST-2 binary sentiment analysis dataset using FlashTopic control tokens.
    """
    
    def __init__(self, model_id_or_path: str, cache_dir: str = None, 
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8000):
        """
        Initialize the SST-2 FlashTopic evaluator.
        
        Args:
            model_id_or_path: Path to the model or model ID
            cache_dir: Directory for caching models and tokenizers
            gpu_memory_utilization: GPU memory utilization ratio for vLLM
            max_model_len: Maximum sequence length for the model
        """
        self.model_id_or_path = model_id_or_path
        self.cache_dir = cache_dir
        
        # Load model and tokenizer with vLLM
        print(f"Loading finetuned model {model_id_or_path} with vLLM...")
        
        # Initialize vLLM
        self.llm = LLM(
            model=model_id_or_path,
            tokenizer=model_id_or_path,
            dtype="bfloat16",  # Use bfloat16 for memory efficiency
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            # download_dir=cache_dir,
            quantization="fp8",
        )
        
        # Load processor using AutoProcessor
        print("Loading processor with AutoProcessor...")
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path, 
            cache_dir=cache_dir, 
            trust_remote_code=True
        )
        
        # Set up tokenizer from processor
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        
        # Add control tokens for sentiment categories
        control_tokens = ["[control_1]", "[control_2]"]
        existing_tokens = []
        for token in control_tokens:
            try:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                if token_id != self.processor.tokenizer.unk_token_id:
                    existing_tokens.append(token)
            except:
                pass
        
        if existing_tokens:
            print(f"Found {len(existing_tokens)} existing control tokens in model")
        else:
            print("No existing control tokens found, adding them...")
            n_new = self.processor.tokenizer.add_special_tokens({"additional_special_tokens": control_tokens})
            print(f"Added {n_new} control tokens to tokenizer")
        
        # Create control token mappings for sentiment categories
        self.sentiment_tokens = {
            'Negative': '[control_1]',
            'Positive': '[control_2]'
        }
        
        print(f"Model loaded successfully with vLLM control token support!")
    
    def build_sentiment_prompt(self, text: str) -> str:
        """
        Build prompt for SST-2 binary sentiment analysis with control tokens.
        
        Args:
            text: The movie review text
            
        Returns:
            Formatted prompt string with control tokens
        """
        # Build the prompt with control tokens directly replacing sentiment labels
        prompt = f"""Please analyze the sentiment of the following movie review and classify it as:

[control_1] Negative
[control_2] Positive

Review: {text}

Based on the overall sentiment expressed in this review, respond with the relevant control token:"""
        
        return prompt
    
    def predict_sentiment(self, text: str) -> Tuple[str, float, Optional[Dict]]:
        """
        Predict the sentiment for a given movie review using control tokens.
        
        Args:
            text: The movie review text
            
        Returns:
            Tuple of (predicted_sentiment, inference_latency, logprobs)
        """
        # Build the prompt with control tokens
        prompt = self.build_sentiment_prompt(text)
        
        # Create messages for chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template using processor
        prompt_token_ids = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            top_p=1.0,
            top_k=1000,
            max_tokens=1,  # Generate only one token (the control token)
            logprobs=20,  # Get logprobs for top 20 tokens
            stop=["<end_of_turn>", "<eos>", "\n"]  # Stop at end tokens or newline
        )
        
        # START TIMING
        start_time = time.time()
        
        # Generate response using vLLM
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = self.llm.generate(
                [{"prompt_token_ids": prompt_token_ids}],
                sampling_params
            )
            generated_tokens = outputs[0].outputs[0].token_ids
            response = self.processor.tokenizer.decode(generated_tokens).strip()
            
            # Get logprobs for analysis if available
            logprobs = outputs[0].outputs[0].logprobs[0] if hasattr(outputs[0].outputs[0], 'logprobs') else None
        
        # END TIMING
        end_time = time.time()
        inference_latency = end_time - start_time
        
        return response, inference_latency, logprobs
    
    def map_prediction_to_label(self, prediction: str, logprobs=None) -> str:
        """
        Map model prediction to sentiment label using logprobs analysis.
        """
        if not logprobs:
            return "unknown"
        
        # Build label_tokens with control token format for classification
        label_tokens = ["[control_1]", "[control_2]"]
        
        # Extract rank and logprob information for control tokens
        rank_dict, logprobs_dict = {}, {}
        for token_id, info in logprobs.items():
            t = info.decoded_token
            if t in label_tokens:
                rank_dict[t] = info.rank
                logprobs_dict[t] = info.logprob
        
        # If we found control tokens in logprobs, use them to determine sentiment
        if rank_dict:
            result = self.postprocess_single_answer(rank_dict, label_tokens)
            return result if result is not None else "unknown"
        
        return "unknown"

    def postprocess_single_answer(self, rank_dict, label_tokens):
        """
        Postprocess to get single best answer based on token ranks.
        """
        if not rank_dict:
            return None
        
        try:
            # Find the token with the lowest rank (highest probability)
            best_token = min(rank_dict, key=rank_dict.get)
            
            # Extract the number from control token format [control_X]
            if best_token.startswith("[control_"):
                try:
                    idx = int(best_token.replace("[control_", "").replace("]", "")) - 1
                    if 0 <= idx < 2:  # We have 2 sentiment categories
                        return ['Negative', 'Positive'][idx]
                except ValueError:
                    pass
            
            return None
        except (KeyError, ValueError, IndexError):
            return None
    
    def test_sentiment_classification(self):
        """
        Test sentiment classification functionality to ensure it works properly.
        """
        print("\n🔍 Testing sentiment classification functionality...")
        
        # Test a simple prediction to see sentiment classification in action
        print("Testing sentiment classification with sample reviews...")
        
        # Test positive review
        positive_review = "This movie is absolutely fantastic! I loved every minute of it."
        result_pos, latency_pos, logprobs_pos = self.predict_sentiment(positive_review)
        extracted_pos = self.map_prediction_to_label(result_pos, logprobs_pos)
        print(f"Positive review test:")
        print(f"  Review: '{positive_review}'")
        print(f"  Raw Response: '{result_pos}'")
        print(f"  Extracted Sentiment: '{extracted_pos}'")
        print(f"  Latency: {latency_pos:.2f}s")
        
        # Test negative review
        negative_review = "This movie is terrible. I hated it completely."
        result_neg, latency_neg, logprobs_neg = self.predict_sentiment(negative_review)
        extracted_neg = self.map_prediction_to_label(result_neg, logprobs_neg)
        print(f"Negative review test:")
        print(f"  Review: '{negative_review}'")
        print(f"  Raw Response: '{result_neg}'")
        print(f"  Extracted Sentiment: '{extracted_neg}'")
        print(f"  Latency: {latency_neg:.2f}s")
    
    def evaluate_dataset(self, data_loader: SST2DataLoader, max_samples: int = None, split: str = 'validation') -> Dict:
        """
        Evaluate the model on the dataset.
        
        Args:
            data_loader: SST2DataLoader instance
            max_samples: Maximum number of samples to evaluate (None for all)
            split: Split to evaluate ('train', 'validation', or 'test')
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        print(f"Starting evaluation on SST-2 {split} split (binary classification with control tokens)...")
        
        # Get the appropriate data split
        if split == 'train':
            data = data_loader.train_data
        elif split == 'validation':
            data = data_loader.validation_data
        elif split == 'test':
            data = data_loader.test_data
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Initialize result containers
        predictions = []
        ground_truth = []
        prediction_texts = []
        latencies = []
        correct_predictions = []
        wrong_predictions = []
        unknown_predictions = []
        logprobs_data = []
        
        # Determine number of samples to evaluate
        total_samples = len(data)
        if max_samples:
            total_samples = min(total_samples, max_samples)
        
        print(f"Evaluating {total_samples} samples...")
        
        # Process each sample
        for idx in tqdm(range(total_samples), desc="Processing samples"):
            sample = data[idx]
            if not sample:
                continue
            
            # Extract sample information
            text = sample['text']
            true_label = sample['label']
            true_label_id = sample['label_id']
            
            # Predict sentiment
            pred_text, inference_latency, logprobs = self.predict_sentiment(text)
            predicted_label = self.map_prediction_to_label(pred_text, logprobs)
            
            # Store results
            predictions.append(predicted_label)
            ground_truth.append(true_label)
            prediction_texts.append(pred_text)
            latencies.append(inference_latency)
            logprobs_data.append(logprobs)
            
            # Track prediction types
            if predicted_label == 'unknown':
                unknown_predictions.append((idx, text[:100], pred_text))
            elif predicted_label == true_label:
                correct_predictions.append((idx, text[:100], predicted_label, true_label, true_label_id))
            else:
                wrong_predictions.append((idx, text[:100], predicted_label, true_label, true_label_id))
        
        # Calculate metrics
        total = len(predictions)
        correct_count = len(correct_predictions)
        unknown_count = len(unknown_predictions)
        wrong_count = len(wrong_predictions)
        
        accuracy = correct_count / total if total > 0 else 0
        
        # Calculate latency statistics
        p50_latency = np.percentile(latencies, 50) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        mean_latency = np.mean(latencies) if latencies else 0
        
        # Calculate per-class metrics
        class_report = classification_report(ground_truth, predictions, output_dict=True, zero_division=0)
        
        # Compile results
        results = {
            'dataset': f'SST-2_{split}',
            'total_samples': total,
            'accuracy': accuracy,
            'correct_predictions': correct_count,
            'wrong_predictions': wrong_count,
            'unknown_predictions': unknown_count,
            'predictions': predictions,
            'ground_truth': ground_truth,
            'prediction_texts': prediction_texts,
            'latencies': latencies,
            'p50_latency_sec': p50_latency,
            'p95_latency_sec': p95_latency,
            'mean_latency_sec': mean_latency,
            'correct_samples': correct_predictions,
            'wrong_samples': wrong_predictions,
            'unknown_samples': unknown_predictions,
            'classification_report': class_report,
            'logprobs_data': logprobs_data
        }
        
        # Print evaluation summary
        print(f"\n" + "="*60)
        print(f"EVALUATION SUMMARY - SST-2 {split.upper()} (Control Tokens)")
        print(f"="*60)
        print(f"Total samples: {total}")
        print(f"Correct predictions: {correct_count} ({correct_count/total*100:.2f}%)")
        print(f"Wrong predictions: {wrong_count} ({wrong_count/total*100:.2f}%)")
        print(f"Unknown predictions: {unknown_count} ({unknown_count/total*100:.2f}%)")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"P50 latency: {p50_latency:.2f} sec")
        print(f"P95 latency: {p95_latency:.2f} sec")
        print(f"Mean latency: {mean_latency:.2f} sec")
        
        # Print per-class results
        print(f"\nPer-Class Results:")
        for class_name in data_loader.class_labels:
            if class_name in class_report:
                metrics = class_report[class_name]
                print(f"  {class_name}: Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
        
        return results


def main():
    """
    Main function to run SST-2 evaluation with vLLM and control tokens.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Evaluate model on SST-2 dataset with vLLM and control tokens')
    parser.add_argument('--model_path', type=str, 
                       default='./merged_multimodal_mintrec',
                       help='Path to trained model')
    parser.add_argument('--cache_dir', type=str, default="./hf_cache",
                       help='Cache directory for models')
    parser.add_argument('--output_dir', type=str, default='sst2_vllm_flashtopic_results',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to evaluate')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                       help='GPU memory utilization for vLLM (default: 0.9)')
    parser.add_argument('--max_model_len', type=int, default=8000,
                       help='Maximum model length for vLLM (default: 8000)')
    
    args = parser.parse_args()
    
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, f'sst2_vllm_flashtopic_evaluation_{args.split}.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load dataset
    logging.info(f"Loading SST-2 dataset from Hugging Face...")
    data_loader = SST2DataLoader()
    
    # Initialize evaluator
    logging.info("Initializing SST-2 FlashTopic evaluator with vLLM...")
    evaluator = SST2FlashTopicEvaluator(
        args.model_path, 
        args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Test the model with a simple example
    print("\n�� Testing model with example query...")
    test_sample = data_loader.get_sample(0, args.split)
    if test_sample:
        print(f"Test Review: {test_sample['text'][:100]}...")
        print(f"True Sentiment: {test_sample['label']} (Label ID: {test_sample['label_id']})")
        
        result, latency, logprobs = evaluator.predict_sentiment(test_sample['text'])
        predicted_label = evaluator.map_prediction_to_label(result, logprobs)
        print(f"Prediction: Raw: '{result}' → Extracted: '{predicted_label}' (Latency: {latency:.2f}s)")
    
    # Test sentiment classification functionality
    evaluator.test_sentiment_classification()
    
    # Run evaluation
    logging.info(f"Starting SST-2 evaluation on {args.split} split...")
    results = evaluator.evaluate_dataset(data_loader, max_samples=args.max_samples, split=args.split)
    
    # Save results to JSON file
    output_file = os.path.join(args.output_dir, f'sst2_vllm_flashtopic_results_{args.split}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {output_file}")
    
    # Print final evaluation summary
    print("\n" + "="*60)
    print(f"FINAL EVALUATION SUMMARY - SST-2 {args.split.upper()} (Control Tokens)")
    print("="*60)
    print(f"Dataset: {results['dataset']}")
    print(f"Total samples: {results['total_samples']}")
    print(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"Correct: {results['correct_predictions']}")
    print(f"Wrong: {results['wrong_predictions']}")
    print(f"Unknown: {results['unknown_predictions']}")
    print(f"P50 latency: {results['p50_latency_sec']:.2f} sec")
    print(f"P95 latency: {results['p95_latency_sec']:.2f} sec")
    print(f"Mean latency: {results['mean_latency_sec']:.2f} sec")
    
    # Save detailed results
    detailed_file = os.path.join(args.output_dir, f'sst2_vllm_flashtopic_detailed_{args.split}.txt')
    with open(detailed_file, 'w') as f:
        f.write(f"SST-2 vLLM FlashTopic Evaluation Results - {args.split.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Correct predictions: {results['correct_predictions']}\n")
        f.write(f"Wrong predictions: {results['wrong_predictions']}\n")
        f.write(f"Unknown predictions: {results['unknown_predictions']}\n\n")
        
        f.write("WRONG PREDICTIONS:\n")
        f.write("-" * 40 + "\n")
        for idx, text, pred, true_label, label_id in results['wrong_samples'][:20]:  # Show first 20
            f.write(f"Sample {idx}: Predicted {pred}, Correct {true_label} (Label ID: {label_id})\n")
            f.write(f"Review: {text[:100]}...\n\n")
        
        f.write("UNKNOWN PREDICTIONS:\n")
        f.write("-" * 40 + "\n")
        for idx, text, pred_text in results['unknown_samples'][:20]:  # Show first 20
            f.write(f"Sample {idx}: Raw prediction: {pred_text[:100]}...\n")
            f.write(f"Review: {text[:100]}...\n\n")
    
    logging.info(f"Detailed results saved to {detailed_file}")


if __name__ == "__main__":
    main()