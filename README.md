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
| Script | Description | Model Support | Link | Local Path |
|--------|-------------|---------------|------|----------------|
| **`sft_trainer_gemma3_from_json.py`** | Fine-tuning script for Gemma 3 models | Gemma 3 (4B/27B) | [🔗 View Script](./code/training/sft_trainer_gemma3_from_json.py) | `./code/training/`<br>`sft_trainer_gemma3_from_json.py` |
| **`sft_trainer_mistral_from_json.py`** | Fine-tuning script for Mistral models | Mistral (24B) | [🔗 View Script](./code/training/sft_trainer_mistral_from_json.py) | `./code/training/`<br>`sft_trainer_mistral_from_json.py` |
| **`merge_mistral.py`** | LoRA weight merging script for Mistral3 | Mistral (24B) | [🔗 View Script](./code/training/merge_mistral.py) | `./code/training/`<br>`merge_mistral.py` |

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
| Script | Purpose | Dataset | Link | Local Path |
|--------|---------|---------|------|----------------|
| **`mintrec_evaluation_vllm_flashtopic.py`** | Comprehensive evaluation using vLLM | MIntRec 2.0 | [🔗 View Script](./code/evaluation/mintrec_evaluation_vllm_flashtopic.py) | `./code/evaluation/`<br>`mintrec_evaluation_vllm_flashtopic.py` |
| **`agnews_evaluation_vllm_flashtopic.py`** | AG News classification evaluation | AG News | [🔗 View Script](./code/evaluation/agnews_evaluation_vllm_flashtopic.py) | `./code/evaluation/`<br>`agnews_evaluation_vllm_flashtopic.py` |
| **`amazon_reviews_evaluation_vllm_flashtopic.py`** | Amazon Reviews sentiment evaluation | Amazon Reviews | [🔗 View Script](./code/evaluation/amazon_reviews_evaluation_vllm_flashtopic.py) | `./code/evaluation/`<br>`amazon_reviews_evaluation_vllm_flashtopic.py` |
| **`dbpedia_evaluation_vllm_flashtopic.py`** | DBpedia classification evaluation | DBpedia | [🔗 View Script](./code/evaluation/dbpedia_evaluation_vllm_flashtopic.py) | `./code/evaluation/`<br>`dbpedia_evaluation_vllm_flashtopic.py` |
| **`sst2_evaluation_vllm_flashtopic.py`** | SST-2 sentiment evaluation | SST-2 | [🔗 View Script](./code/evaluation/sst2_evaluation_vllm_flashtopic.py) | `./code/evaluation/`<br>`sst2_evaluation_vllm_flashtopic.py` |

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

## 📈 Performance Results


### 🧠 Multimodal Topic Classification (MIntRec 2.0)

<div align="center">

| Model | Modalities | Accuracy | P50 Latency | P95 Latency | Status |
|-------|------------|----------|-------------|-------------|---------|
| **Gemma3 4B** | Text/Image/Video | 16.04% | 1.33s | 1.77s | 🟡 Base |
| **Gemma3 27B** | Text/Image/Video | 17.76% | 2.18s | 2.77s | 🟡 Base |
| **GPT-4o** | Text/Image/Video | 43.68% | 4.30s | 6.12s | 🟡 External |
| **GPT-5-nano** | Text/Image/Video | 41.02% | 3.67s | 5.46s | 🟡 External |
| **GPT-5** | Text/Image/Video | 51.84% | 7.13s | 13.01s | 🟡 External |
| **Mistral3 24B (FT multimodal+text)** | Text/Image/Video | 43.09% | 0.64s | 1.64s | 🟢 Fine-tuned |
| **Gemma3 4B (FT text)** | Text/Image/Video | 25.09% | 0.25s | 0.59s | 🟢 Fine-tuned |
| **Gemma3 4B (FT multimodal)** | Text/Image/Video | **55.19%** | 0.26s | 0.60s | 🟢 Fine-tuned |
| **Gemma3 4B (FT multimodal+text)** | Text/Image/Video | 32.37% | 0.26s | 0.59s | 🟢 Fine-tuned |
| **Gemma3 12B (FT multimodal+text)** | Text/Image/Video | 35.42% | 0.28s | 0.71s | 🟢 Fine-tuned |
| **Gemma3 27B (FT multimodal)** | Text/Image/Video | **62.72%** | 0.37s | 0.90s | 🟢 Fine-tuned |
| **Gemma3 27B (FT multimodal+text)** | Text/Image/Video | 44.27% | 0.36s | 0.88s | 🟢 Fine-tuned |

</div>


## 🏷️ Topic Classification Categories

### 📰 AG News Topics (4 types)
- `World` - World news and international events
- `Sports` - Sports news and athletic events
- `Business` - Business and financial news
- `Science/Technology` - Science and technology news

### 📝 Sentiment Analysis Topics (2 types)
- `Positive` - Positive sentiment reviews
- `Negative` - Negative sentiment reviews

### 🏛️ DBpedia Topics (14 types)
- `Company` - Corporate entities
- `EducationalInstitution` - Schools and universities
- `Artist` - Creative individuals
- `Athlete` - Sports professionals
- `OfficeHolder` - Political officials
- `MeanOfTransportation` - Vehicles and transport
- `Building` - Architectural structures
- `NaturalPlace` - Geographic locations
- `Village` - Small settlements
- `Animal` - Living creatures
- `Plant` - Botanical entities
- `Album` - Music collections
- `Film` - Cinematic works
- `WrittenWork` - Literary pieces

### 🎯 Intent Recognition Topics (30 types)
- `Acknowledge`, `Advise`, `Agree`, `Apologise`, `Arrange`
- `Ask for help`, `Asking for opinions`, `Care`, `Comfort`, `Complain`
- `Confirm`, `Criticize`, `Doubt`, `Emphasize`, `Explain`
- `Flaunt`, `Greet`, `Inform`, `Introduce`, `Invite`
- `Joke`, `Leave`, `Oppose`, `Plan`, `Praise`
- `Prevent`, `Refuse`, `Taunt`, `Thank`, `Warn`

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

