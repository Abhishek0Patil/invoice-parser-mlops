# train.py (version with safetensors fix)

import json
import logging
from pathlib import Path

import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Processor,
    Trainer,
    TrainingArguments,
)
from PIL import Image

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Path Definitions ---
try:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    ROOT_DIR = Path.cwd()

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_DIR = ROOT_DIR / "data" / "dataset"
MODEL_OUTPUT_DIR = ROOT_DIR / "models" / "layoutlmv2-finetuned-sroie"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Model & Processor Loading ---
MODEL_CHECKPOINT = "microsoft/layoutlmv2-base-uncased"
PROCESSOR = LayoutLMv2Processor.from_pretrained(MODEL_CHECKPOINT, apply_ocr=False)

# --- Load Pre-processed Dataset ---
try:
    dataset = load_from_disk(PROCESSED_DATA_DIR / "sroie_dataset_for_layoutlm")
    logging.info(f"Dataset loaded successfully:\n{dataset}")
except FileNotFoundError:
    logging.error(f"FATAL: Dataset not found at {PROCESSED_DATA_DIR / 'sroie_dataset_for_layoutlm'}")
    exit()


# --- Label Definitions ---
labels = dataset['train'].features['ner_tags'].feature.names
id2label = {i: label for i, label in enumerate(labels)}
label2id = {label: i for i, label in enumerate(labels)}
logging.info(f"Labels for model: {label2id}")


def normalize_box(box, width, height):
    """Normalizes a bounding box to a [0, 1000] range."""
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

def preprocess_data(examples):
    """
    Preprocesses a batch of examples, now with explicit bounding box normalization.
    """
    # 1. Load images to get their dimensions
    image_paths = [RAW_DATA_DIR / "img" / f"{id}.jpg" for id in examples['id']]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    
    # 2. Normalize bounding boxes for each image in the batch
    normalized_bboxes = []
    for i, (image, bboxes) in enumerate(zip(images, examples['bboxes'])):
        width, height = image.size
        normalized_bboxes.append([normalize_box(box, width, height) for box in bboxes])

    # 3. Pass everything to the processor
    encoded_inputs = PROCESSOR(
        images, 
        examples['words'], 
        boxes=normalized_bboxes,  # Use the newly normalized boxes
        word_labels=examples['ner_tags'],
        padding="max_length", 
        truncation=True,
    )
    return encoded_inputs

# --- Metrics Calculation ---
metric = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"], "recall": results["overall_recall"],
        "f1": results["overall_f1"], "accuracy": results["overall_accuracy"],
    }

def main():
    logging.info("Starting data preprocessing...")
    processed_dataset = dataset.map(
        preprocess_data, batched=True, remove_columns=dataset['train'].column_names,
    )
    logging.info(f"Dataset after processing:\n{processed_dataset}")

    # --- Model Initialization ---
    model = LayoutLMv2ForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True  # <--- THIS IS THE FIX
    )
    
    # --- Training Arguments (AGGRESSIVELY OPTIMIZED for 4GB VRAM) ---
    training_args = TrainingArguments(
        output_dir=str(MODEL_OUTPUT_DIR),
        num_train_epochs=20,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        fp16=True,
        learning_rate=3e-5,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
    )

    # --- Trainer Initialization ---
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        compute_metrics=compute_metrics,
    )
    
    # --- Start Training ---
    logging.info("Starting model training...")
    trainer.train()
    
    logging.info("Training complete. Saving the best model and processor...")
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    PROCESSOR.save_pretrained(str(MODEL_OUTPUT_DIR))
    
    logging.info("Evaluating on the held-out test set...")
    test_results = trainer.evaluate(processed_dataset["test"])
    
    logging.info(f"Test Set Evaluation Results: {test_results}")
    with open(MODEL_OUTPUT_DIR / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=4)
    
    logging.info(f"Script finished. Best model saved to {MODEL_OUTPUT_DIR}")

if __name__ == '__main__':
    main()