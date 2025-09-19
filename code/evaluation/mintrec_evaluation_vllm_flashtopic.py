"""
python3 mintrec_evaluation_vllm_flashtopic.py \
  --data_path ./MIntRec2.0/ \
  --model_path ./your_finetuned_model \
  --video_data_path ./MIntRec2.0/in-scope/video/ \
  --use_video 
"""
# Standard library imports for file operations, data processing, and system utilities
import os
import csv
import json
import pickle
import numpy as np
import torch
import requests
from PIL import Image
import cv2
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
from transformers import AutoTokenizer, AutoProcessor
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')
import subprocess
import shutil
from decord import VideoReader
import time
import multiprocessing as mp

# vLLM specific imports and configuration
# Set multiprocessing method to spawn for vLLM compatibility
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
mp.set_start_method("spawn", force=True)
from vllm import LLM, SamplingParams

class MIntRec2DataLoader:
    """
    Data loader for MIntRec2.0 dataset.
    
    This class handles loading and preprocessing the MIntRec2.0 dataset,
    which contains multimodal conversation data with intent labels.
    """
    
    def __init__(self, data_path: str, dataset: str = 'MIntRec2.0'):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the MIntRec2.0 dataset directory
            dataset: Dataset name (default: 'MIntRec2.0')
        """
        self.data_path = data_path
        self.dataset = dataset
        
        # Define the 30 intent labels for MIntRec2.0 dataset
        self.intent_labels = [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn'
        ]
        
        # Create bidirectional mappings between labels and IDs
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Load all data splits (train, dev, test)
        self.train_data = self._load_split('train')
        self.dev_data = self._load_split('dev')
        self.test_data = self._load_split('test')
        
        print(f"Loaded {len(self.train_data)} train, {len(self.dev_data)} dev, {len(self.test_data)} test samples")
    
    def _load_split(self, split: str) -> List[Dict]:
        """
        Load data from a specific split (train/dev/test).
        
        Args:
            split: Split name ('train', 'dev', or 'test')
            
        Returns:
            List of dictionaries containing sample data
        """
        data = []
        tsv_path = os.path.join(self.data_path, 'in-scope', f'{split}.tsv')
        
        # Check if the split file exists
        if not os.path.exists(tsv_path):
            print(f"Warning: {tsv_path} not found. Skipping {split} split.")
            return data
            
        # Read TSV file and parse each row
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)  # Skip header row
            
            for row in reader:
                if len(row) >= 4:  # Ensure we have enough columns
                    # Generate unique index: dia{dialogue_id}_utt{utterance_id}
                    index = '_'.join(['dia' + str(row[0]), 'utt' + str(row[1])])
                    
                    # Extract text utterance and intent label from the row
                    text_utterance = row[2]
                    intent_label = row[3]
                    
                    # Create sample dictionary with all necessary information
                    sample = {
                        'id': index,  # Unique identifier for the sample
                        'text': text_utterance,  # Text utterance from the conversation
                        'intent': intent_label,  # Ground truth intent label
                        'intent_id': self.label_to_id.get(intent_label, -1),  # Numeric intent ID
                        'dialogue_id': row[0],  # Original dialogue ID
                        'utterance_id': row[1]  # Original utterance ID
                    }
                    data.append(sample)
        
        return data
    
    def get_split(self, split: str) -> List[Dict]:
        """
        Get data from specified split.
        
        Args:
            split: Split name ('train', 'dev', or 'test')
            
        Returns:
            List of sample dictionaries for the requested split
            
        Raises:
            ValueError: If split name is not recognized
        """
        if split == 'train':
            return self.train_data
        elif split == 'dev':
            return self.dev_data
        elif split == 'test':
            return self.test_data
        else:
            raise ValueError(f"Unknown split: {split}")


def extract_video_frames(video_path, output_dir, fps=1, max_duration=15):
    """
    Extract frames from video file using either ffmpeg or opencv as fallback.
    
    This function extracts video frames at specified FPS and saves them as JPEG images.
    It first tries to use ffmpeg for better performance, then falls back to opencv/decord.
    
    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted frames
        fps: Frames per second to extract (default: 1)
        max_duration: Maximum duration in seconds to process (default: 15)
        
    Returns:
        List of paths to extracted frame images
    """
    import subprocess
    import shutil
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Check if frames already exist to avoid re-extraction
    existing_frames = [f for f in os.listdir(frames_dir) if f.endswith('.jpg')]
    if existing_frames:
        frame_paths = [os.path.join(frames_dir, f) for f in sorted(existing_frames)]
        return frame_paths
    
    # Try ffmpeg first for better performance
    if shutil.which("ffmpeg"):
        frames_cmd = [
            "ffmpeg", "-i", video_path, 
            "-vf", f"fps={fps}", 
            "-t", str(max_duration),
            f"{frames_dir}/%04d.jpg",
            "-y"  # Overwrite existing files
        ]
        
        try:
            subprocess.run(frames_cmd, check=True, capture_output=True)
            frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
            frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
            return frame_paths
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg failed for {video_path}: {e}")
    
    # Fallback to opencv/decord if ffmpeg fails
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        duration = len(vr)
        fps_video = vr.get_avg_fps()
        
        # Calculate frame indices to extract based on desired FPS
        total_seconds = min(duration / fps_video, max_duration)
        num_frames = int(total_seconds * fps)
        frame_indices = [int(i * duration / num_frames) for i in range(num_frames)]
        
        # Extract frames and convert to numpy arrays
        frames = vr.get_batch(frame_indices).asnumpy()
        frame_paths = []
        
        # Save each frame as JPEG image
        for i, frame in enumerate(frames):
            frame_path = os.path.join(frames_dir, f"{i+1:04d}.jpg")
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)
        
        return frame_paths
        
    except Exception as e:
        print(f"Error processing video {video_path} with opencv/decord: {e}")
        return []


def get_placeholders_for_videos(frames: List, timestamps=[]):
    """
    Create content placeholders for video frames following Aria format.
    
    This function creates the proper content structure for multimodal prompts
    that include video frames, following the Aria model's expected format.
    
    Args:
        frames: List of video frames (PIL Images)
        timestamps: Optional list of timestamps for each frame
        
    Returns:
        List of content placeholders for the multimodal prompt
    """
    contents = []
    if not timestamps:
        # Simple format: just image placeholders followed by newline
        for i, _ in enumerate(frames):
            contents.append({"text": None, "type": "image"})
        contents.append({"text": "\n", "type": "text"})
    else:
        # Format with timestamps: [MM:SS] + image + newline for each frame
        for i, (_, ts) in enumerate(zip(frames, timestamps)):
            contents.extend([
                {"text": f"[{int(ts)//60:02d}:{int(ts)%60:02d}]", "type": "text"},
                {"text": None, "type": "image"},
                {"text": "\n", "type": "text"}
            ])
    return contents


class MultimodalIntentEvaluator:
    """
    Evaluator for multimodal models on intent recognition task using vLLM.
    
    This class provides comprehensive evaluation capabilities for multimodal models
    on the MIntRec2.0 intent recognition dataset, supporting both text-only and
    multimodal (text + video) evaluation.
    """
    
    def __init__(self, model_id_or_path: str, cache_dir: str = None, 
                 gpu_memory_utilization: float = 0.9, max_model_len: int = 8000):
        """
        Initialize the multimodal evaluator.
        
        Args:
            model_id_or_path: Path to the multimodal model or model ID
            cache_dir: Directory for caching models and tokenizers
            gpu_memory_utilization: GPU memory utilization ratio for vLLM
            max_model_len: Maximum sequence length for the model
        """
        self.model_id_or_path = model_id_or_path
        self.cache_dir = cache_dir
        
        # Load model and tokenizer with vLLM
        print(f"Loading multimodal model {model_id_or_path} with vLLM...")
        
        # Initialize vLLM with multimodal support (generic configuration)
        self.llm = LLM(
            model=model_id_or_path,
            tokenizer=model_id_or_path,
            dtype="bfloat16",  # Use bfloat16 for memory efficiency
            limit_mm_per_prompt={"image": 64},  
            enforce_eager=True,
            trust_remote_code=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            download_dir=cache_dir,
        )
        
        # Load processor using AutoProcessor like in training
        print("Loading processor with AutoProcessor...")
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path, 
            cache_dir=cache_dir, 
            trust_remote_code=True
        )
        
        # Set up tokenizer from processor (like in training)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"
        
        # Add control tokens if they exist in the model
        control_tokens = [f"[control_{i}]" for i in range(1, 31)]  # 1-30
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
        
        print(f"Model loaded successfully with vLLM multimodal support!")

        # Define the 30 intent labels for MIntRec2.0 dataset
        self.intent_labels = [
            'Acknowledge', 'Advise', 'Agree', 'Apologise', 'Arrange', 
            'Ask for help', 'Asking for opinions', 'Care', 'Comfort', 'Complain', 
            'Confirm', 'Criticize', 'Doubt', 'Emphasize', 'Explain', 
            'Flaunt', 'Greet', 'Inform', 'Introduce', 'Invite', 
            'Joke', 'Leave', 'Oppose', 'Plan', 'Praise', 
            'Prevent', 'Refuse', 'Taunt', 'Thank', 'Warn'
        ]
        
        # Create bidirectional mappings between labels and IDs
        self.label_to_id = {label: idx for idx, label in enumerate(self.intent_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}
        
        # Create control tokens for each intent label (format: [control_1], [control_2], etc.)
        self.label_tokens = [f"[control_{i+1}]" for i in range(len(self.intent_labels))]

    
    def build_intent_prompt(self, text: str) -> str:
        """
        Build intent classification prompt aligned with training format.
        
        This method creates a prompt that matches the format used during model training,
        using control tokens to guide the model's intent classification.
        
        Args:
            text: Input text to classify
            
        Returns:
            List of messages in the format expected by the chat template
        """
        
        # Build the topics list with control tokens (exactly like training format)
        topics_list = []
        for idx, label in enumerate(self.intent_labels):
            control_token = self.label_tokens[idx]
            topics_list.append(f"{control_token} {label}\n#####")
        
        topics_text = "\n".join(topics_list)
        
        # Create the system prompt exactly like training format
        system_prompt = f"""You are a topic classification expert. Before making a decision, carefully follow all the topic-specific instructions/descriptions.
Topics:
{topics_text}"""
        
        # Create user message with the conversation text
        user_message = f"### USER CONVERSATION HERE ###\n{text}\n\nBased on the above conversation, respond with the relevant topic ID:\n"
        
        # Create messages in the correct format for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return messages
    
    def predict_intent_text_only(self, text: str) -> Tuple[str, float, Optional[Dict]]:
        """
        Predict intent using text only with control token prompt template.
        
        This method performs intent classification using only the text modality,
        following the control token approach used during training.
        
        Args:
            text: Input text to classify
            
        Returns:
            Tuple of (predicted_text, inference_latency, logprobs)
        """
        # Build the prompt with control tokens
        messages = self.build_intent_prompt(text)
        
        # Apply chat template using processor (like in training)
        prompt_token_ids = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        print(self.processor.tokenizer.decode(prompt_token_ids, skip_special_tokens=False))
        
        # Configure sampling parameters for deterministic generation
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic generation
            top_p=1.0,
            top_k=1000,
            max_tokens=1,  # Generate only one token (the control token)
            logprobs=20,  # Get logprobs for top 20 tokens (vLLM limitation)
            stop=["<end_of_turn>", "<eos>"]  # Stop at end tokens
        )
        
        # START TIMING: Only measure the actual model inference
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
            
        # END TIMING: Only the model inference time
        end_time = time.time()
        inference_latency = end_time - start_time
        
        print(f"Text-only response: {response}")
        
        return response, inference_latency, logprobs
    
    def predict_intent_multimodal(self, text: str, video_path: str, 
                                 cache_dir: str = "video_cache") -> Tuple[str, float, Optional[Dict]]:
        """
        Predict intent using both text and video with control token prompt template.
        
        This method performs intent classification using both text and video modalities.
        It extracts frames from the video and combines them with the text input
        for multimodal intent recognition.
        
        Args:
            text: Input text to classify
            video_path: Path to the video file
            cache_dir: Directory for caching extracted video frames
            
        Returns:
            Tuple of (predicted_text, inference_latency, logprobs)
        """
        try:
            # Extract frames from video
            video_basename = os.path.basename(video_path).replace('.mp4', '')
            video_cache_dir = os.path.join(cache_dir, video_basename)
            frame_paths = extract_video_frames(video_path, video_cache_dir, fps=1, max_duration=15)
            
            # Fallback to text-only if no frames extracted
            if not frame_paths:
                print(f"No frames extracted from {video_path}, falling back to text-only")
                return self.predict_intent_text_only(text)
            
            # Limit number of frames to prevent memory issues
            max_frames = min(len(frame_paths), 8)
            frame_paths_to_use = frame_paths[:max_frames]
            
            # Load images from frame paths
            frames = []
            for frame_path in frame_paths_to_use:
                try:
                    img = Image.open(frame_path).convert("RGB")
                    frames.append(img)
                except Exception as e:
                    print(f"Error loading frame {frame_path}: {e}")
                    continue
            
            # Fallback to text-only if no valid images loaded
            if not frames:
                print(f"No valid images loaded from {video_path}, falling back to text-only")
                return self.predict_intent_text_only(text)
            
            # Build the prompt with control tokens (EXACTLY like training format)
            messages = self.build_intent_prompt(text)
            
            # Create user content EXACTLY like training format
            user_query = f"### USER CONVERSATION HERE ###\n{text}\n\nBased on the above conversation, respond with the relevant topic ID:\n"
            
            # Use get_placeholders_for_videos to create proper content placeholders
            contents = get_placeholders_for_videos(frames)
            
            # Create user content with placeholders + text (EXACTLY like training)
            user_content = contents + [{"type": "text", "text": user_query}]
            
            # Create messages in EXACT training format
            messages = [
                {"role": "system", "content": messages[0]["content"]},  # System prompt with topics
                {"role": "user", "content": user_content}  # Video placeholders + text query
            ]
            print("Multimodal messages (training format):", messages)
            
            try:
                # Apply chat template using processor (like in training)
                prompt_token_ids = self.processor.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
                
                # Configure sampling parameters for multimodal generation
                sampling_params = SamplingParams(
                    temperature=0.0,  # Deterministic generation
                    top_p=1.0,
                    top_k=1000,
                    max_tokens=1,  # Generate only one token (the control token)
                    logprobs=20,  # Get logprobs for top 20 tokens
                    stop=["<end_of_turn>", "<eos>"]  # Stop at end tokens
                )
                
                # START TIMING
                start_time = time.time()
                
                # Use exact training format for multimodal generation
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    outputs = self.llm.generate(
                        [{"prompt_token_ids": prompt_token_ids, "multi_modal_data": {"image": frames}}],
                        sampling_params
                    )
                    generated_tokens = outputs[0].outputs[0].token_ids
                    response = self.processor.tokenizer.decode(generated_tokens).strip()
                    
                    # Get logprobs for analysis if available
                    logprobs = outputs[0].outputs[0].logprobs[0] if hasattr(outputs[0].outputs[0], 'logprobs') else None
                
                end_time = time.time()
                inference_latency = end_time - start_time
                
                print(f"Multimodal response: {response}")
                
            except Exception as e:
                print(f"Multimodal generation failed: {e}")
                return self.predict_intent_text_only(text)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            return self.predict_intent_text_only(text)
        
        return response, inference_latency, logprobs
    
    def postprocess_single_intent(self, rank_dict, label_tokens):
        """
        Postprocess to get single best intent based on token ranks.
        
        This method selects the intent with the highest rank (lowest rank number)
        from the control tokens.
        
        Args:
            rank_dict: Dictionary mapping tokens to their ranks
            label_tokens: List of control tokens for intent labels
            
        Returns:
            Best intent label based on token ranks, or None if no valid mapping found
        """
        if not rank_dict:
            return None  # Return None instead of default intent
        
        try:
            # Find the token with the lowest rank (highest probability)
            best_token = min(rank_dict, key=rank_dict.get)
            
            # Extract the number from control token format [control_X]
            if best_token.startswith("[control_"):
                try:
                    idx = int(best_token.replace("[control_", "").replace("]", "")) - 1
                    if 0 <= idx < len(self.intent_labels):
                        return self.intent_labels[idx]
                except ValueError:
                    pass
            
            # Return None if no valid mapping found
            return None
        except (KeyError, ValueError, IndexError):
            return None


    def map_prediction_to_label(self, prediction: str, logprobs=None) -> str:
        """
        Map model prediction to intent label using only logprobs analysis.
        
        This method uses logprobs to determine the most likely intent label
        by analyzing the probability distribution over control tokens.
        
        Args:
            prediction: Raw model prediction text (not used in this implementation)
            logprobs: Logprobs for advanced analysis (required)
            
        Returns:
            Mapped intent label based on logprobs analysis, or "unknown" if no valid mapping found
        """
        # If no logprobs available, return "unknown"
        if not logprobs:
            return "unknown"
        
        # Build label_tokens with control token format
        label_tokens = [f"[control_{i+1}]" for i in range(len(self.intent_labels))]
        
        # Extract rank and logprob information for control tokens
        rank_dict, logprobs_dict = {}, {}
        for token_id, info in logprobs.items():
            t = info.decoded_token
            if t in label_tokens:
                rank_dict[t] = info.rank
                logprobs_dict[t] = info.logprob
        
        # If we found control tokens in logprobs, use them to determine intent
        if rank_dict:
            result = self.postprocess_single_intent(rank_dict, label_tokens)
            return result if result is not None else "unknown"
        
        # If no control tokens found in logprobs, return "unknown"
        return "unknown"
    
    def evaluate_split(self, data: List[Dict], split_name: str, use_video: bool = False, 
                      video_data_path: str = None) -> Dict:
        """
        Evaluate model on a specific data split.
        
        This method performs comprehensive evaluation on a dataset split,
        including accuracy, latency, and detailed classification metrics.
        
        Args:
            data: List of data samples to evaluate
            split_name: Name of the split being evaluated
            use_video: Whether to use video modality
            video_data_path: Path to video data directory
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        print(f"Evaluating on {split_name} split...")
        
        # Initialize result containers
        predictions = []
        ground_truth = []
        prediction_texts = []
        latencies = []
        wrong_predictions = []
        
        print(f"\nStarting {split_name} evaluation...")

        # Process each sample in the split
        for idx, sample in enumerate(tqdm(data, desc=f"Processing {split_name}")):
            print(f"\nProcessing index: {idx}")
            text = sample['text']
            true_intent = sample['intent']
            ground_truth.append(true_intent)

            # Choose prediction method based on video availability
            if use_video and video_data_path:
                video_path = os.path.join(video_data_path, f"{sample['id']}.mp4")
                if os.path.exists(video_path):
                    pred_text, inference_latency, logprobs = self.predict_intent_multimodal(text, video_path)
                else:
                    pred_text, inference_latency, logprobs = self.predict_intent_text_only(text)
            else:
                pred_text, inference_latency, logprobs = self.predict_intent_text_only(text)

            # Process the prediction using logprobs for better accuracy
            predicted_label = self.map_prediction_to_label(pred_text, logprobs)
            predictions.append(predicted_label)
            prediction_texts.append(pred_text)
            latencies.append(inference_latency)  # Use pure inference latency

            # Track incorrect predictions for analysis (unknown is considered incorrect)
            if predicted_label == "unknown" or true_intent.lower() != predicted_label.lower():
                wrong_predictions.append((text, true_intent, predicted_label))
        
        # Calculate final metrics (unknown predictions are considered incorrect)
        total = len(data)
        top1_correct = sum(1 for gt, pred in zip(ground_truth, predictions) 
                       if gt.lower() == pred.lower())
        top1_accuracy = top1_correct / total if total else 0
        
        # Calculate latency statistics
        p50_latency = np.percentile(latencies, 50) if latencies else 0
        p95_latency = np.percentile(latencies, 95) if latencies else 0
        mean_latency = np.mean(latencies) if latencies else 0
        
        # Generate detailed classification metrics (Include unknown as a special class)
        # Add "unknown" to labels if it's not already there
        all_labels = self.intent_labels + ["unknown"]
        
        try:
            report = classification_report(ground_truth, predictions, output_dict=True)
            cm = confusion_matrix(ground_truth, predictions, labels=all_labels)
        except Exception as e:
            print(f"Warning: Could not generate classification report: {e}")
            report = {"accuracy": 0.0}
            cm = np.zeros((len(all_labels), len(all_labels)))
        
        # Compile comprehensive results
        results = {
            'split': split_name,
            'accuracy': top1_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': predictions,
            'ground_truth': ground_truth,
            'prediction_texts': prediction_texts,
            'num_samples': total,
            'latencies': latencies,
            'p50_latency_sec': p50_latency,
            'p95_latency_sec': p95_latency,
            'mean_latency_sec': mean_latency,
            'wrong_predictions': wrong_predictions,
            'top1_correct': top1_correct,
            'total': total,
            'unknown_predictions': sum(1 for pred in predictions if pred == "unknown"),  # Track unknown predictions
            'valid_predictions': sum(1 for pred in predictions if pred != "unknown"),  # Track valid predictions
        }
        
        # Print evaluation summary
        print(f"\nTop-1 Accuracy: {top1_accuracy:.2%} ({top1_correct}/{total})")
        print(f"Unknown predictions: {results['unknown_predictions']}/{total} ({results['unknown_predictions']/total*100:.1f}%)")
        print(f"Valid predictions: {results['valid_predictions']}/{total} ({results['valid_predictions']/total*100:.1f}%)")
        print(f"P50 latency: {p50_latency:.2f} sec")
        print(f"P95 latency: {p95_latency:.2f} sec")
        print(f"Mean latency: {mean_latency:.2f} sec")
        
        # Print confusion matrix summary
        unknown_count = results['unknown_predictions']
        correct_count = top1_correct
        incorrect_count = total - unknown_count - correct_count
        
        print(f"\nPrediction Breakdown:")
        print(f"  Correct: {correct_count} ({correct_count/total*100:.1f}%)")
        print(f"  Incorrect: {incorrect_count} ({incorrect_count/total*100:.1f}%)")
        print(f"  Unknown: {unknown_count} ({unknown_count/total*100:.1f}%)")
        
        try:
            print(f"\nClassification Report:\n{classification_report(ground_truth, predictions)}")
        except Exception as e:
            print(f"Could not print classification report: {e}")
        
        return results
    
    def evaluate_all_splits(self, data_loader: MIntRec2DataLoader, use_video: bool = False, 
                           video_data_path: str = None) -> Dict:
        """
        Evaluate on all splits.
        
        This method orchestrates evaluation across all dataset splits,
        currently focusing on the test split.
        
        Args:
            data_loader: DataLoader instance containing the dataset
            use_video: Whether to use video modality
            video_data_path: Path to video data directory
            
        Returns:
            Dictionary containing results for all evaluated splits
        """
        all_results = {}
        
        # Evaluate on test set
        test_data = data_loader.get_split('test')
        test_results = self.evaluate_split(test_data, 'test', use_video, video_data_path)
        all_results['test'] = test_results
        
        return all_results

def main():
    """
    Main function to run the multimodal model evaluation on MIntRec2.0 dataset.
    
    This function sets up the evaluation pipeline, including argument parsing,
    model initialization, testing, and comprehensive evaluation.
    """
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description='Evaluate multimodal model on MIntRec2.0 dataset with vLLM')
    parser.add_argument('--data_path', type=str, required=True, 
                       help='Path to MIntRec2.0 dataset')
    parser.add_argument('--video_data_path', type=str, default=None,
                       help='Path to video data (optional)')
    parser.add_argument('--model_path', type=str, 
                   default='./merged_multimodal_mintrec',  # Use trained multimodal model
                   help='Path to trained multimodal model')
    parser.add_argument('--cache_dir', type=str, default="./hf_cache",
                       help='Cache directory for models')
    parser.add_argument('--output_dir', type=str, default='multimodal_results_vllm',
                       help='Output directory for results')
    parser.add_argument('--use_video', action='store_true',
                       help='Use video modality in addition to text')
    parser.add_argument('--max_frames', type=int, default=8,
                       help='Maximum number of frames to extract per video (default: 8)')
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
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Load dataset
    logging.info("Loading MIntRec2.0 dataset...")
    data_loader = MIntRec2DataLoader(args.data_path)
    
    # Initialize evaluator
    logging.info("Initializing multimodal evaluator with vLLM...")
    evaluator = MultimodalIntentEvaluator(
        args.model_path, 
        args.cache_dir,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len
    )
    
    # Test the model with simple examples
    print("\n🔵 Testing multimodal model with example queries...")
    test_query = "Hello, how are you doing today?"
    result, latency, logprobs = evaluator.predict_intent_text_only(test_query)
    mapped_result = evaluator.map_prediction_to_label(result, logprobs)
    print(f"Test Query: '{test_query}' → Raw Prediction: '{result}' → Mapped: '{mapped_result}' (Latency: {latency:.2f}s)")
    
    test_query2 = "Thank you so much for your help!"
    result2, latency2, logprobs = evaluator.predict_intent_text_only(test_query2)
    mapped_result2 = evaluator.map_prediction_to_label(result2, logprobs)
    print(f"Test Query: '{test_query2}' → Raw Prediction: '{result2}' → Mapped: '{mapped_result2}' (Latency: {latency2:.2f}s)")
    
    #  Add here: Specific control token testing
    print("\n🔍 Testing control token generation specifically...")
    
    # Test 1: Check control token positions in vocabulary
    print("\nControl token positions in vocabulary:")
    for i in [1, 15, 29]:  # Test several control tokens
        token = f"[control_{i}]"
        try:
            token_id = evaluator.processor.tokenizer.convert_tokens_to_ids(token)
            print(f"  {token} → ID: {token_id}")
            if token_id == evaluator.processor.tokenizer.unk_token_id:
                print(f"    ⚠️  WARNING: {token} maps to UNK token!")
        except Exception as e:
            print(f"    ❌ Error with {token}: {e}")
    
    # Run evaluation
    logging.info("Starting evaluation...")
    results = evaluator.evaluate_all_splits(
        data_loader, 
        use_video=args.use_video, 
        video_data_path=args.video_data_path
    )
    
    # Save results to JSON file
    output_file = os.path.join(args.output_dir, 'evaluation_results_vllm.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {output_file}")
    
    # Print final evaluation summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY (Multimodal Model vLLM)")
    print("="*50)
    for split, result in results.items():
        print(f"{split.upper()} SET:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Samples: {result['num_samples']}")
        print(f"  P50 latency: {result['p50_latency_sec']:.2f} sec")
        print(f"  P95 latency: {result['p95_latency_sec']:.2f} sec")
        print(f"  Mean latency: {result['mean_latency_sec']:.2f} sec")
        print()


if __name__ == "__main__":
    main()
