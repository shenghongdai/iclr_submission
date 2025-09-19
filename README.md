# 🚀 Multimodal Topic Classification Fine-tuning


## 🎯 Project Overview

This project provides a complete pipeline for fine-tuning multimodal models (Gemma 3, Mistral) on various datasets for topic classification. Our models can classify conversation topics from both **text** and **image/video** inputs, achieving state-of-the-art performance on multimodal intent recognition tasks.

### ✨ Key Features
- 🎥 **Multimodal Support**: Text, image, and video processing capabilities
- 🚀 **High Performance**: Optimized training with DeepSpeed and LoRA
- 📊 **Comprehensive Evaluation**: Multiple evaluation frameworks and metrics
- 🔧 **Easy Deployment**: vLLM integration for fast inference
- 🌍 **Multilingual**: Support for 140+ languages

---

## 📁 Project Structure

### 🎯 Core Training Scripts

```
code/
└── training/
    ├── sft_trainer_gemma3_from_json.py    # Fine-tuning script for Gemma 3 models (4B/27B)
    ├── sft_trainer_mistral_from_json.py   # Fine-tuning script for Mistral models (24B)
    └── merge_mistral.py                   # LoRA weight merging script for Mistral3
```

> 💡 **Note**: Both scripts feature a custom `build_collator` for image processing adapted from [TRL's sft_vlm_gemma3.py](https://github.com/huggingface/trl/blob/v0.21.0/examples/scripts/sft_vlm_gemma3.py)

### 🔄 Key Differences: Mistral3 vs Gemma3 Finetuning

#### **Mistral3 (Untied Architecture)**
- **Weight Tying**: ❌ **No weight tying** - `embed_tokens` and `lm_head` are separate
- **LoRA Config**: Uses `modules_to_save=["embed_tokens", "lm_head"]` to make embeddings trainable
- **Merge Step**: ⚠️ **Requires separate merge step** using `merge_mistral.py` after training
- **Final Model**: LoRA weights remain separate, need explicit merging

#### **Gemma3 (Tied Architecture)**  
- **Weight Tying**: ✅ **Uses weight tying** - `model.tie_weights()` connects embeddings
- **LoRA Config**: No `modules_to_save`, embeddings automatically trainable via weight tying
- **Merge Step**: ✅ **Automatic merging** during training with `trainer.model.merge_and_unload()`
- **Final Model**: LoRA weights automatically merged into base model

> 🚨 **Important**: Mistral3 users must run the merge step separately after training to combine LoRA weights with the base model for inference.

### 📊 Evaluation Scripts

```
code/
└── evaluation/
    ├── mintrec_evaluation_vllm_flashtopic.py      # Comprehensive evaluation using vLLM (MIntRec 2.0)
    ├── agnews_evaluation_vllm_flashtopic.py       # AG News classification evaluation
    ├── amazon_reviews_evaluation_vllm_flashtopic.py  # Amazon Reviews sentiment evaluation
    ├── dbpedia_evaluation_vllm_flashtopic.py      # DBpedia classification evaluation
    └── sst2_evaluation_vllm_flashtopic.py         # SST-2 sentiment evaluation
```

### 🔧 Data Processing Scripts

```
code/
└── data_processing/
    └── vlm_data/
```

---

## 🤖 Supported Models

<div align="center">

| Model | Parameters | Modalities | Performance |
|-------|------------|------------|-------------|
| [`google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it) | **4B** | Text/Image/Video | ⭐⭐⭐⭐ |
| [`google/gemma-3-27b-it`](https://huggingface.co/google/gemma-3-27b-it) | **27B** | Text/Image/Video | ⭐⭐⭐⭐⭐ |
| [`mistralai/Mistral-Small-3.1-24B-Instruct-2503`](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) | **24B** | Text/Image/Video | ⭐⭐⭐⭐ |

</div>

---

## 📊 Training Datasets

This project supports both **multimodal datasets** and **text-only classification datasets** for comprehensive topic classification training. See [DATA.md](vlm_data/README.md) for complete details.

### 🎥 **Multimodal Datasets** 

<div align="center">

| Dataset | Year | Modalities | License | Notes |
|---------|------|------------|---------|-------|
| **MIntRec** | 2022 | Text + Video + Audio | [MIT](https://github.com/thuiar/MIntRec/blob/main/LICENSE) | First multimodal dialogue intent dataset ([GitHub](https://github.com/thuiar/MIntRec)) |
| **A-OKVQA** | 2021 | Image + Text | [Apache-2.0](https://github.com/allenai/aokvqa/blob/main/LICENSE) | ~25K commonsense VQA pairs requiring world knowledge ([GitHub](https://github.com/allenai/aokvqa)) |
| **Visual7W** | 2016 | Image + Text + BBoxes | [MIT](https://github.com/yukezhu/visual7w-toolkit/blob/master/LICENSE) | 327K 7W questions + object groundings ([GitHub](https://github.com/yukezhu/visual7w-toolkit)) |

</div>

---


## 🚀 Quick Start

### 1️⃣ Training 
#### **Gemma3 Training**

```bash
# For Gemma3 4B model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file ./configs/multi_gpu_deepspeed3.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  sft_trainer_gemma3_from_json.py \
  --model_name google/gemma-3-4b-it \
  --json_path ./vlm_data/combined_control_updated_nested.json \
  --output_dir ./runs_sft_gemma3_4b_full_fast

# For Gemma3 27B model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file ./configs/multi_gpu_deepspeed3.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  sft_trainer_gemma3_from_json.py \
  --model_name google/gemma-3-27b-it \
  --json_path ./vlm_data/combined_control_updated_nested.json \
  --output_dir ./runs_sft_gemma3_27b_full_fast \
  --grad_accum 8
```

#### **Mistral3 Training + Merge Step**

```bash
# Step 1: Training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file ./configs/multi_gpu_deepspeed3.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  sft_trainer_mistral_from_json.py \
  --json_path ./vlm_data/combined_control_updated_nested.json \
  --output_dir ./runs_sft_mistral3_24b_full_fast

# Step 2: Merge LoRA weights (REQUIRED for Mistral3)
python3 merge_mistral.py
```

> ⚠️ **Critical for Mistral3**: The merge step is mandatory after training to combine LoRA weights with the base model. Without merging, the model won't work for inference.

### 📁 **Fine-tuned Model Locations**

Trained models are saved in the following local directories:

| Model | Local Path |
|-------|------------|
| **Gemma3 4B** | `./runs_sft_gemma3_4b_full_fast/final_checkpoint` |
| **Gemma3 27B** | `./runs_sft_gemma3_27b_full_fast/final_checkpoint` |
| **Mistral3 24B** | `./merged_mistral24b_full_fast/` |

> 💡 **Note**: Use these paths when running evaluation scripts or deploying models for inference.

### 2️⃣ Evaluation

#### 🔍 Option 1: MIntRec 2.0 Evaluation
```bash
python3 mintrec_evaluation_vllm_flashtopic.py \
  --data_path ./MIntRec2.0/ \
  --model_path ./your_finetuned_model \
  --video_data_path ./MIntRec2.0/in-scope/video/ \
  --use_video 
```

#### 📰 Option 2: AG News Classification Evaluation
```bash
python3 agnews_evaluation_vllm_flashtopic.py \
  --data_path ./ag_news_data \
  --model_path ./your_finetuned_model
```

#### 📝 Option 3: Amazon Reviews Sentiment Evaluation
```bash
python3 amazon_reviews_evaluation_vllm_flashtopic.py \
  --model_path ./your_finetuned_model
```

#### 🏛️ Option 4: DBpedia Classification Evaluation
```bash
python3 dbpedia_evaluation_vllm_flashtopic.py \
  --model_path ./your_finetuned_model
```

#### 📊 Option 5: SST-2 Sentiment Evaluation
```bash
python3 sst2_evaluation_vllm_flashtopic.py \
  --model_path ./your_finetuned_model
```

> ⚠️ **Note**: Update `MODEL_PATH` in the scripts before running

---


## 📋 Requirements & Setup

### 🔧 Dependencies
```bash
# Core ML libraries
pip install torch transformers peft accelerate datasets

# Multimodal processing
pip install vllm decord opencv-python pillow

# Training optimization
pip install trl bitsandbytes
```

### ⚙️ Configuration Options

#### 🎯 Training Hyperparameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_name` | `google/gemma-3-4b-it` | Base model to fine-tune (e.g., `google/gemma-3-27b-it`) |
| `--learning_rate` | 2e-5 | Learning rate for training |
| `--num_epochs` | 30 | Number of training epochs |
| `--batch_size` | 1 | Training batch size |
| `--grad_accum` | 16 | Gradient accumulation steps |

#### 🔗 LoRA Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lora_r` | 8 | LoRA rank parameter |
| `--lora_alpha` | 16 | LoRA alpha parameter |

#### 📊 Evaluation Configuration
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--eval_steps` | 500 | Steps between evaluations |
| `--eval_strategy` | "steps" | Evaluation strategy |

---

## 🚨 Important Notes

### ⚠️ Memory Optimization
- **Reduce `--grad_accum` from 16 to 8** if you encounter OOM issues
- **Monitor GPU memory** during training

### 🛑 Early Stopping
Early stopping parameters are critical for optimal training:
- `early_stopping_patience=8` - stops training if no improvement for 8 evaluations
- `early_stopping_threshold=1e-4` - minimum improvement threshold
- **Adjust these based on your dataset size and training stability**

### 🔄 Mistral3 Merge Requirements
- **Mistral3 requires explicit LoRA merging** after training using `merge_mistral.py`
- **Without merging, the model cannot be used for inference**
- **The merge step combines LoRA weights with the base model**
- **Update paths in `merge_mistral.py` before running** (BASE_MODEL, PEFT_DIR, OUT_DIR)

---

## 📊 Key Insights

### ✅ **Strengths**
- **High Accuracy**: Models achieve excellent performance on text classification tasks
- **Fast Processing**: Under 0.2 seconds for most models
- **Scalable**: 27B models handle complex tasks with high accuracy
- **Multimodal Support**: Effective text and image processing capabilities

### ⚠️ **Limitations**
- **4B Model Complexity**: Smaller models may struggle with very complex multimodal tasks
- **Multimodal Training**: Requires careful balance between modalities
- **Dataset Dependency**: Performance varies based on training data quality and diversity

---

