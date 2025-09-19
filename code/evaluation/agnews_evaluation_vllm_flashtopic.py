#!/usr/bin/env python3
"""
AG News Evaluation Script for Finetuned Models using vLLM (Custom Control Tokens)

This script evaluates a finetuned model on the AG News dataset using control tokens,
which contains news articles categorized into 4 classes:
World, Sports, Business, Science/Technology.
"""

import os
import csv
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

class AGNewsDataLoader:
    """
    Data loader for AG News dataset.
    
    This class handles loading and preprocessing the AG News dataset,
    which contains news articles for text classification.
    """
    
    def __init__(self, data_path: str, dataset_name: str = 'ag_news'):
        """
        Initialize the AG News data loader.
        
        Args:
            data_path: Path to the AG News dataset directory
            dataset_name: Name of the dataset file
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        
        # Define the 4 classes for AG News dataset
        self.class_labels = ['World', 'Sports', 'Business', 'Science/Technology']
        self.label_to_id = {label: idx for idx, label in enumerate(self.class_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Load the dataset
        self.train_data = self._load_split('train')
        self.test_data = self._load_split('test')
        
        print(f"Loaded {len(self.train_data)} train, {len(self.test_data)} test samples from {dataset_name}")
    
    def _load_split(self, split: str) -> List[Dict]:
        """
        Load data from a specific split (train/test).
        
        Args:
            split: Split name ('train' or 'test')
            
        Returns:
            List of dictionaries containing sample data
        """
        data = []
        csv_path = os.path.join(self.data_path, f'{split}.csv')
        
        # Check if the split file exists
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found. Skipping {split} split.")
            return data
            
        # Read CSV file and parse each row
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader):
                try:
                    if len(row) < 2:
                        continue
                        
                    # AG News format: label,title,description
                    label_id = int(row[0]) - 1  # Convert to 0-based indexing
                    title = row[1] if len(row) > 1 else ''
                    description = row[2] if len(row) > 2 else ''
                    
                    # Combine title and description for full text
                    full_text = f"{title}. {description}".strip()
                    
                    sample = {
                        'index': row_num,
                        'label_id': label_id,
                        'label': self.id_to_label.get(label_id, 'Unknown'),
                        'title': title,
                        'description': description,
                        'text': full_text,
                        'split': split
                    }
                    data.append(sample)
                    
                except Exception as e:
                    print(f"Warning: row {row_num} parse error: {e}")
                    continue
        
        return data
    
    def get_sample(self, idx: int, split: str = 'test') -> Optional[Dict]:
        """
        Get a specific sample by index.
        
        Args:
            idx: Sample index
            split: Split to get sample from ('train' or 'test')
            
        Returns:
            Sample dictionary or None if index out of range
        """
        data = self.test_data if split == 'test' else self.train_data
        if 0 <= idx < len(data):
            return data[idx]
        return None


class AGNewsEvaluator:
    """
    Evaluator for finetuned models on AG News dataset using vLLM with control tokens.
    
    This class provides comprehensive evaluation capabilities for finetuned models
    on the AG News text classification dataset using control tokens for structured output.
    """
    
    def __init__(self, model_id_or_path: str, cache_dir: str = None, 
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8000):
        """
        Initialize the AG News evaluator.
        
        Args:
            model_id_or_path: Path to the finetuned model or model ID
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
        
        # Add control tokens for classification categories
        control_tokens = ["[control_1]", "[control_2]", "[control_3]", "[control_4]"]
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
        
        # Create control token mappings for categories
        self.category_tokens = {
            'World': '[control_1]',
            'Sports': '[control_2]', 
            'Business': '[control_3]',
            'Science/Technology': '[control_4]'
        }
        
        print(f"Model loaded successfully with vLLM control token support!")
    
    def build_agnews_prompt(self, text: str) -> str:
        """
        Build prompt for AG News text classification with control tokens.
        
        Args:
            text: The news article text
            
        Returns:
            Formatted prompt string with control tokens
        """
        # Build the prompt with control tokens directly replacing category names
        prompt = f"""Please classify the following news article into one of these categories:

[control_1] World
[control_2] Sports
[control_3] Business
[control_4] Science/Technology

Article: {text}

Based on the content of this article, respond with the relevant control token:"""
        
        return prompt
    
    def predict_class(self, text: str) -> Tuple[str, float, Optional[Dict]]:
        """
        Predict the class for a given news article using control tokens.
        
        Args:
            text: The news article text
            
        Returns:
            Tuple of (predicted_class, inference_latency, logprobs)
        """
        # Build the prompt with control tokens
        prompt = self.build_agnews_prompt(text)
        
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
        Map model prediction to category label using logprobs analysis.
        
        This method uses logprobs to determine the most likely category
        by analyzing the probability distribution over control tokens.
        
        Args:
            prediction: Raw model prediction text (not used in this implementation)
            logprobs: Logprobs for advanced analysis (required)
            
        Returns:
            Mapped category label based on logprobs analysis, or "unknown" if no valid mapping found
        """
        # If no logprobs available, return "unknown"
        if not logprobs:
            return "unknown"
        
        # Build label_tokens with control token format for classification
        label_tokens = ["[control_1]", "[control_2]", "[control_3]", "[control_4]"]
        
        # Extract rank and logprob information for control tokens
        rank_dict, logprobs_dict = {}, {}
        for token_id, info in logprobs.items():
            t = info.decoded_token
            if t in label_tokens:
                rank_dict[t] = info.rank
                logprobs_dict[t] = info.logprob
        
        # If we found control tokens in logprobs, use them to determine category
        if rank_dict:
            result = self.postprocess_single_answer(rank_dict, label_tokens)
            return result if result is not None else "unknown"
        
        # If no control tokens found in logprobs, return "unknown"
        return "unknown"
    
    def postprocess_single_answer(self, rank_dict, label_tokens):
        """
        Postprocess to get single best answer based on token ranks.
        
        This method selects the category with the highest rank (lowest rank number)
        from the control tokens.
        
        Args:
            rank_dict: Dictionary mapping tokens to their ranks
            label_tokens: List of control tokens for category labels
            
        Returns:
            Best category label based on token ranks, or None if no valid mapping found
        """
        if not rank_dict:
            return None  # Return None instead of default answer
        
        try:
            # Find the token with the lowest rank (highest probability)
            best_token = min(rank_dict, key=rank_dict.get)
            
            # Extract the number from control token format [control_X]
            if best_token.startswith("[control_"):
                try:
                    idx = int(best_token.replace("[control_", "").replace("]", "")) - 1
                    if 0 <= idx < 4:  # We have 4 categories
                        return ['World', 'Sports', 'Business', 'Science/Technology'][idx]
                except ValueError:
                    pass
            
            # Return None if no valid mapping found
            return None
        except (KeyError, ValueError, IndexError):
            return None
    
    def test_control_tokens(self):
        """
        Test control token functionality to ensure they are properly loaded.
        """
        print("\n🔍 Testing control token functionality...")
        
        # Test control token positions in vocabulary
        print("Control token positions in vocabulary:")
        for category, token in self.category_tokens.items():
            try:
                token_id = self.processor.tokenizer.convert_tokens_to_ids(token)
                print(f"  {token} → ID: {token_id}")
                if token_id == self.processor.tokenizer.unk_token_id:
                    print(f"    ⚠️  WARNING: {token} maps to UNK token!")
                else:
                    print(f"    ✅ {token} is properly loaded")
            except Exception as e:
                print(f"    ❌ Error with {token}: {e}")
        
        # Test a simple prediction to see control tokens in action
        print("\nTesting control token prediction with sample article...")
        test_article = "The stock market reached new highs today as technology companies led the rally."
        
        try:
            # Test control token prediction
            result, latency, logprobs = self.predict_class(test_article)
            extracted = self.map_prediction_to_label(result, logprobs)
            print(f"Control token test:")
            print(f"  Article: '{test_article[:50]}...'")
            print(f"  Raw Response: '{result}'")
            print(f"  Extracted Category: '{extracted}'")
            print(f"  Latency: {latency:.2f}s")
            
        except Exception as e:
            print(f"Control token test failed: {e}")
    
    def evaluate_dataset(self, data_loader: AGNewsDataLoader, max_samples: int = None, split: str = 'test') -> Dict:
        """
        Evaluate the model on the dataset.
        
        Args:
            data_loader: AGNewsDataLoader instance
            max_samples: Maximum number of samples to evaluate (None for all)
            split: Split to evaluate ('train' or 'test')
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        print(f"Starting evaluation on AG News {split} split...")
        
        # Get the appropriate data split
        data = data_loader.test_data if split == 'test' else data_loader.train_data
        
        # Initialize result containers
        predictions = []
        ground_truth = []
        prediction_texts = []
        latencies = []
        correct_predictions = []
        wrong_predictions = []
        unknown_predictions = []
        
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
            
            # Predict class using control tokens
            pred_text, inference_latency, logprobs = self.predict_class(text)
            predicted_label = self.map_prediction_to_label(pred_text, logprobs)
            
            # Store results
            predictions.append(predicted_label)
            ground_truth.append(true_label)
            prediction_texts.append(pred_text)
            latencies.append(inference_latency)
            
            # Track prediction types
            if predicted_label == 'unknown':
                unknown_predictions.append((idx, text[:100], pred_text))
            elif predicted_label == true_label:
                correct_predictions.append((idx, text[:100], predicted_label, true_label))
            else:
                wrong_predictions.append((idx, text[:100], predicted_label, true_label))
        
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
        class_report = classification_report(ground_truth, predictions, output_dict=True)
        
        # Compile results
        results = {
            'dataset': f'AG_News_{split}',
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
            'classification_report': class_report
        }
        
        # Print evaluation summary
        print(f"\n" + "="*60)
        print(f"EVALUATION SUMMARY - AG News {split.upper()}")
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
    Main function to run AG News evaluation with control tokens.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Evaluate finetuned model on AG News dataset with vLLM (Control Tokens)')
    parser.add_argument('--data_path', type=str, default='./ag_news_data',
                       help='Path to AG News dataset directory')
    parser.add_argument('--model_path', type=str, 
                       default='./merged_multimodal_mintrec',
                       help='Path to trained finetuned model')
    parser.add_argument('--cache_dir', type=str, default="./hf_cache",
                       help='Cache directory for models')
    parser.add_argument('--output_dir', type=str, default='agnews_results_vllm_control_tokens',
                       help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test'],
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
            logging.FileHandler(os.path.join(args.output_dir, f'agnews_evaluation_control_tokens_{args.split}.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load dataset
    logging.info(f"Loading AG News dataset...")
    data_loader = AGNewsDataLoader(args.data_path)
    
    # Initialize evaluator
    logging.info("Initializing AG News evaluator with vLLM (Control Tokens)...")
    evaluator = AGNewsEvaluator(
        args.model_path, 
        args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Test the model with a simple example
    print("\n�� Testing model with example query...")
    test_sample = data_loader.get_sample(0, args.split)
    if test_sample:
        print(f"Test Article: {test_sample['text'][:100]}...")
        print(f"True Label: {test_sample['label']}")
        
        result, latency, logprobs = evaluator.predict_class(test_sample['text'])
        predicted_label = evaluator.map_prediction_to_label(result, logprobs)
        print(f"Prediction: Raw: '{result}' → Extracted: '{predicted_label}' (Latency: {latency:.2f}s)")
    
    # Test control token functionality
    evaluator.test_control_tokens()
    
    # Run evaluation
    logging.info(f"Starting AG News evaluation on {args.split} split...")
    results = evaluator.evaluate_dataset(data_loader, max_samples=args.max_samples, split=args.split)
    
    # Save results to JSON file
    output_file = os.path.join(args.output_dir, f'agnews_results_control_tokens_{args.split}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {output_file}")
    
    # Print final evaluation summary
    print("\n" + "="*60)
    print(f"FINAL EVALUATION SUMMARY - AG News {args.split.upper()}")
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
    detailed_file = os.path.join(args.output_dir, f'agnews_detailed_control_tokens_{args.split}.txt')
    with open(detailed_file, 'w') as f:
        f.write(f"AG News Evaluation Results (Control Tokens) - {args.split.upper()}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total samples: {results['total_samples']}\n")
        f.write(f"Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Correct predictions: {results['correct_predictions']}\n")
        f.write(f"Wrong predictions: {results['wrong_predictions']}\n")
        f.write(f"Unknown predictions: {results['unknown_predictions']}\n\n")
        
        f.write("WRONG PREDICTIONS:\n")
        f.write("-" * 40 + "\n")
        for idx, text, pred, true_label in results['wrong_samples'][:20]:  # Show first 20
            f.write(f"Sample {idx}: Predicted {pred}, Correct {true_label}\n")
            f.write(f"Text: {text[:100]}...\n\n")
        
        f.write("UNKNOWN PREDICTIONS:\n")
        f.write("-" * 40 + "\n")
        for idx, text, pred_text in results['unknown_samples'][:20]:  # Show first 20
            f.write(f"Sample {idx}: Raw prediction: {pred_text[:100]}...\n")
            f.write(f"Text: {text[:100]}...\n\n")
    
    logging.info(f"Detailed results saved to {detailed_file}")


if __name__ == "__main__":
    main()