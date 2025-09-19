# 🚀 Multimodal Topic Classification Fine-tuning


## 🎯 Project Overview

This repository contains the official code for our ICLR 2026 submission. We introduce a unified framework that reduces classification to a **single-token generation** problem. By fine-tuning decoder-style LLMs (Gemma 3, Mistral 3) with LoRA and atomic label tokens, our models achieve **O(1) latency** and state-of-the-art accuracy on challenging multimodal benchmarks.

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
    └── merge_mistral.py                   # LoRA weight merging script for Mistral-3
```


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


## 🧪 Results (Highlight)

- **MIntRec 2.0**: Fine-tuned **Gemma-3-27B** achieves **62.7% accuracy** with **P95 < 1s** tail latency.
- Text benchmarks (SST-2, Amazon, AG News, DBpedia): competitive accuracy with **5–8× lower tail latency** than strong APIs.

👉 See the paper for full tables and plots.

---

## 🚀 Quick Start

### 1️⃣ Training 
#### **Gemma-3 Training**

```bash
# For Gemma-3 4B model
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file ./configs/multi_gpu_deepspeed3.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  sft_trainer_gemma3_from_json.py \
  --model_name google/gemma-3-4b-it \
  --json_path ./vlm_data/combined_control_updated_nested.json \
  --output_dir ./runs_sft_gemma3_4b_full_fast

# For Gemma-3 27B model
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

#### **Mistral-3 Training + Merge Step**

```bash
# Step 1: Training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file ./configs/multi_gpu_deepspeed3.yaml \
  --num_processes 8 \
  --main_process_port 29500 \
  sft_trainer_mistral_from_json.py \
  --json_path ./vlm_data/combined_control_updated_nested.json \
  --output_dir ./runs_sft_mistral3_24b_full_fast

# Step 2: Merge LoRA weights (REQUIRED for Mistral-3)
python3 merge_mistral.py
```

> ⚠️ **Critical for Mistral-3**: The merge step is mandatory after training to combine LoRA weights with the base model. Without merging, the model won't work for inference.

### 📁 **Fine-tuned Model Locations**

Trained models are saved in the following local directories:

| Model | Local Path |
|-------|------------|
| **Gemma-3 4B** | `./runs_sft_gemma3_4b_full_fast/final_checkpoint` |
| **Gemma-3 27B** | `./runs_sft_gemma3_27b_full_fast/final_checkpoint` |
| **Mistral-3 24B** | `./merged_mistral24b_full_fast/` |

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

#### 📊 Text Classification Evaluation
```bash
# AG News, Amazon Reviews, DBpedia, SST-2 evaluations
python3 [dataset]_evaluation_vllm_flashtopic.py \
  --model_path ./your_finetuned_model
```

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


