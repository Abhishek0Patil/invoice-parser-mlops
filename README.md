# Invoice & Receipt Parser using LayoutLMv2

An end-to-end machine learning project to automatically extract key information (e.g., company name, date, total amount) from document images. This project leverages the multimodal **LayoutLMv2** model, which understands both the text and the visual layout of a document for high-accuracy parsing.

The primary goal is to build a robust pipeline that can be deployed as a service, demonstrating a full cycle of MLOps from data engineering to model training and inference.


---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [1. Data Preparation & Augmentation](#1-data-preparation--augmentation)
  - [2. OCR & Bounding Box Normalization](#2-ocr--bounding-box-normalization)
  - [3. Model Fine-Tuning on Consumer GPU](#3-model-fine-tuning-on-consumer-gpu)
- [Results & Analysis](#results--analysis)
- [Next Steps & Future Work](#next-steps--future-work)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [How to Run](#how-to-run)
  - [1. (Optional) Data Preparation](#1-optional-data-preparation)
  - [2. Model Training](#2-model-training)
  - [3. Inference](#3-inference)

---

## Project Overview

This project tackles the real-world business problem of automated data entry. By processing an image of a receipt or invoice, the system identifies and extracts relevant entities, returning them in a structured JSON format. This has direct applications in accounting automation, expense management, and digital document archiving.

The problem is framed as a **Token Classification / Named Entity Recognition (NER)** task. Each word detected in the document is classified into predefined categories (`COMPANY`, `DATE`, `TOTAL`, etc.) or marked as irrelevant (`O`).

## Key Features

*   **Multimodal AI:** Utilizes `LayoutLMv2`, which combines text, layout, and visual information for superior accuracy over traditional text-only NER.
*   **End-to-End Pipeline:** Covers the entire ML lifecycle: data collection, OCR, data alignment, model training, and inference.
*   **Optimized for Consumer Hardware:** The training process was successfully executed on an NVIDIA RTX 3050 with 4GB of VRAM by employing memory-saving techniques like mixed-precision (`fp16`) and gradient accumulation.
*   **Extensive Debugging:** The project documents a real-world debugging journey, solving challenges with CUDA versions, `detectron2` compilation on Windows, and data-related errors.
*   **Modern Tooling:** Built with industry-standard libraries including Hugging Face `transformers`, `datasets`, `PyTorch`, and `Tesseract`.

## Tech Stack

*   **Model:** `microsoft/layoutlmv2-base-uncased`
*   **Frameworks:** `PyTorch`, `Hugging Face Transformers`, `Hugging Face Datasets`, `Accelerate`
*   **OCR Engine:** `Tesseract OCR` (via `pytesseract`)
*   **Data Handling:** `pandas`, `Pillow`, `OpenCV`
*   **Metrics & Evaluation:** `seqeval`, `evaluate`

## Project Structure

```
invoice-parser-mlops/
├── data/
│   ├── dataset/              # Raw SROIE dataset images and labels
│   └── processed/            # Pre-processed Hugging Face dataset
├── models/
│   └── layoutlmv2-finetuned-sroie/ # Saved fine-tuned model artifacts
├── src/
│   ├── data_preparation/     # Scripts for data alignment (not included in repo)
│   └── models_code/          # Core model scripts
│       ├── train_model.py
│       └── predict_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Methodology

### 1. Data Preparation & Augmentation
The initial dataset was the public **SROIE dataset** (626 receipts). To enhance model robustness, this was supplemented with invoices found online, bringing the total dataset size to over 1000 documents. All data was split into `train`, `validation`, and `test` sets (80/10/10) before any processing to prevent data leakage.

### 2. OCR & Bounding Box Normalization
1.  **OCR:** The `Tesseract` engine was used to extract all words and their pixel-based bounding boxes from each document image.
2.  **Normalization:** A critical preprocessing step was to normalize the bounding box coordinates. The LayoutLMv2 architecture expects coordinates in a `[0, 1000]` range. A custom function was implemented to scale the pixel coordinates from each image to this required range, preventing `CUDA device-side assert` errors during training.
3.  **BIO Tagging:** A script was used to align the ground-truth labels with the OCR output and assign a `B-I-O` (Beginning, Inside, Outside) tag to each word, preparing the data for the token classification task.

### 3. Model Fine-Tuning on Consumer GPU
The pre-trained `LayoutLMv2` model was fine-tuned on our custom dataset. To overcome the 4GB VRAM limitation of the GPU, the following `TrainingArguments` were essential:
*   `per_device_train_batch_size=1`
*   `gradient_accumulation_steps=8` (achieving an effective batch size of 8)
*   `fp16=True` (mixed-precision training)

---

## Results & Analysis

The model was successfully trained for a total of **20 epochs**.

*   **Final Accuracy:** The model achieved a high overall accuracy of **94%**.
*   **Final Loss:** The evaluation loss consistently decreased, ending at **0.24**, indicating successful learning.
*   **Precision & F1-Score:** The final precision on the test set was **34%**.

**Analysis:** The high accuracy combined with a relatively low precision/F1 score is a classic sign of a class imbalance problem. The model has learned to correctly classify the overwhelmingly common `O` (Outside) tag, but it still struggles to confidently and accurately identify the rarer entity tags (`B-TOTAL`, `B-COMPANY`, etc.).

While the model has learned the foundational aspects of the task, further iteration is required to improve its practical usefulness for entity extraction.

## Next Steps & Future Work

The project is now in the **iterative improvement phase**. The current model serves as a strong baseline.

### Immediate Next Steps
1.  **[In Progress] Data Enrichment:** The highest priority is to significantly increase the size and diversity of the training data. The plan is to add another 200-300 manually-labeled invoices, focusing on layouts and formats not present in the SROIE dataset.
2.  **Hyperparameter Tuning:** After enriching the dataset, the next step is to experiment with different hyperparameters to improve precision. This includes:
    *   Lowering the `learning_rate` (e.g., to `1e-5`).
    *   Introducing `weight_decay` for regularization (e.g., `0.01`).
    *   (Advanced) Experimenting with `class_weights` in the loss function to give more importance to rare entity tags.
3.  **Re-train Final Model:** Train a definitive model from scratch using the enriched dataset and the best-found hyperparameters.

### Future MLOps & Deployment Goals
Once a model with a satisfactory F1-score is achieved, the focus will shift to productionization:
1.  **Build an API Service:** Wrap the inference logic in a **FastAPI** application to create a robust prediction endpoint.
2.  **Containerize with Docker:** Create a `Dockerfile` to package the FastAPI service, model, and all dependencies into a portable container.
3.  **Create a Live Demo:** Develop a simple UI with **Streamlit** or **Gradio** that allows users to upload an invoice and see the results live.
4.  **Deploy to the Cloud:** Host the containerized demo on a platform like **Hugging Face Spaces** to make it publicly accessible and shareable.

---

## Setup and Installation

### Prerequisites
*   Python 3.10+
*   An NVIDIA GPU with CUDA support
*   [Tesseract OCR Engine](https://github.com/tesseract-ocr/tesseract) installed and available in the system's PATH.
*   [Git](https://git-scm.com/)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd invoice-parser-mlops
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv env
    .\env\Scripts\activate
    ```
3.  **Install dependencies:**
    *   First, install PyTorch matching your CUDA version from the [official website](https://pytorch.org/get-started/locally/).
    *   Then, install the remaining packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
    *   Finally, install Detectron2 (required for LayoutLMv2):
    ```bash
    pip install "git+https://github.com/facebookresearch/detectron2.git" --no-build-isolation
    ```

## How to Run

### 1. (Optional) Data Preparation
The repository assumes the processed Hugging Face dataset is already available in `data/processed/`. If you need to regenerate it, you would run your data alignment and preprocessing scripts.

### 2. Model Training
To launch the training process using the parameters defined in the script:
```bash
python src/models_code/train_model.py
```
The script will save the best model and its checkpoints to the `models/layoutlmv2-finetuned-sroie/` directory.

### 3. Inference
To run a prediction on a single test image using the fine-tuned model:
```bash
python src/models_code/predict_model.py
```
Make sure to update the image path inside the script.
