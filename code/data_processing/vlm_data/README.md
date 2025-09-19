# 🚀 VLM Data Preparation Pipeline


> **A comprehensive pipeline for preparing multimodal training data for Vision-Language Models (VLMs)**

This contains a robust and scalable pipeline for processing multiple datasets and converting them into a unified format suitable for fine-tuning advanced models.


## 🌟 Overview

The data preparation pipeline seamlessly combines three major datasets to create a comprehensive training corpus:

- **MIntRec**: Multimodal Intent Recognition dataset with video segments
- **AOKVQA**: Visual Question Answering dataset with COCO images
- **Visual7W**: Visual Question Answering dataset with custom images




## 📊 Data Sources

### 🌐 Local Dataset Placement

These are the **final processed datasets** available locally for immediate training:

#### 1. 🎬 MIntRec Dataset
- `./vlm_data/MIntRec/`

#### 2. 🖼️ AOKVQA Dataset  
- `./vlm_data/coco_images_5k/`

#### 3. 🔍 Visual7W Dataset
- `./vlm_data/visual7w_images_5k/`

---

## 🛠️ Local Processing Pipeline

> **For developers who want to process data locally**

### 📥 Step 1: Download Raw Datasets

#### 🎬 MIntRec Dataset
```bash
# Clone the official repository
git clone https://github.com/thuiar/MIntRec
```

**📁 Expected Structure:**
```
MIntRec/
├── 📁 data/
│   └── 📁 MIntRec/
│       ├── 📁 S04/                    # Season 4 episodes
│       ├── 📁 S05/                    # Season 5 episodes  
│       ├── 📁 S06/                    # Season 6 episodes
│       ├── 📄 train.tsv              # Training data
│       ├── 📄 dev.tsv                 # Development data
│       └── 📄 test.tsv                # Test data
```

#### 🖼️ AOKVQA Dataset
```bash
# Download from official source
git clone https://github.com/allenai/aokvqa
```

**📁 Expected Structure:**
```
aokvqa/
├── 📁 datasets/
│   ├── 📁 coco/
│   │   ├── 📁 train2017/     # 118K images
│   │   └── 📁 annotations/
│   │       ├── 📄 instances_train2017.json
│   │       ├── 📄 captions_train2017.json
│   │       └── 📄 person_keypoints_train2017.json
│   └── 📁 aokvqa/
│       └── 📄 aokvqa_v1p0_train.json
```

#### 🔍 Visual7W Dataset
```bash
# Clone the toolkit
git clone https://github.com/ranjaykrishna/visual7w-toolkit
```

**📁 Expected Structure:**
```
visual7w-toolkit/
├── 📁 images/                 # 47,299+ JPG files
└── 📁 datasets/
    └── 📁 visual7w-telling/
        └── 📄 dataset.json
```

---

## 🔧 Scripts & Tools

### 🎬 Video Frame Extraction

**Script**: `extract_mintrec_frames.py`

Extracts representative frames from MIntRec videos for image-based training.

```bash
python3 extract_mintrec_frames.py --mintrec_dir MIntRec/data/MIntRec
```

### 🔄 Retry Failed Extractions

**Script**: `retry_failed_extractions.py`

Intelligent retry mechanism for failed video frame extractions.

```bash
python3 retry_failed_extractions.py --mintrec_dir MIntRec/data/MIntRec
```

**🔄 Retry Strategies:**
1. **1 Second Timestamp** - Primary extraction point
2. **0.5 Seconds Timestamp** - Fallback extraction point  
3. **First Frame** - Last resort extraction
4. **Codec Variations** - Different ffmpeg settings

### 🎯 Create Unified Dataset

**Script**: `create_image_classification_json.py`

Creates a unified JSON format combining all three datasets with classification tasks.

```bash
python3 create_image_classification_json.py
```

**📊 Sampling Strategy:**
- **AOKVQA**: Sampled to ~5K images for balanced training
- **Visual7W**: Sampled to ~5K images for balanced training
- **MIntRec**: All available frames included

### 📁 Organize Images

**Script**: `extract_and_copy_images.py`

Organizes images into separate folders for easier upload.

```bash
python3 extract_and_copy_images.py
```

**📂 Output Structure:**
```
vlm_data/
├── 📁 coco_images_5k/        # COCO images from AOKVQA
├── 📁 visual7w_images_5k/    # Visual7W custom images
└── 📁 MIntRec/        # MIntRec extracted frames
```

### 🔗 Update Image Paths

**Script**: `update_image_paths.py`

Updates image paths in JSON files to match new folder structures and deployment environments.

```bash
python3 update_image_paths.py
```

### 🎲 Advanced Control Token Management

**Script**: `update_control_tokens_nested_messages.py`

Advanced control token management with nested message structure support.

```bash
python3 update_control_tokens_nested_messages.py
```

**🎯 Features:**
- 🔢 **Random Assignment**: Control tokens 1-500 to prevent overfitting

---

## 📝 Output Formats

### 🎯 Image Classification Format


```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a topic classification expert. Before making a decision, carefully follow all the topic-specific instructions.\n\nTopics:\n[control_x] Complain\n#####\n[control_y] Praise\n#####\n[control_z] Apologise..."
    },
    {
      "role": "user",
      "content": "### USER CONVERSATION HERE ###\n[conversation text]"
    },
    {
      "role": "assistant",
      "content": "[control_x]"
    }
  ],
  "image_path": "path/to/image.jpg",
  "dataset": "MIntRec",
  "intent_label": "Complain",
  "split": "train",
  "correct_answer": "[control_x]",
  "correct_idx": 0
}
```


---

## 🚀 Usage Examples

### 🔥 Complete Pipeline Execution

```bash
# 1. Extract video frames
python3 extract_mintrec_frames.py --mintrec_dir MIntRec/data/MIntRec

# 2. Retry any failed extractions
python3 retry_failed_extractions.py --mintrec_dir MIntRec/data/MIntRec

# 3. Create unified dataset
python3 create_image_classification_json.py

# 4. Organize images
python3 extract_and_copy_images.py

# 5. Update paths for deployment
python3 update_image_paths.py

# 6. Final control token processing
python3 update_control_tokens_nested_messages.py
```

