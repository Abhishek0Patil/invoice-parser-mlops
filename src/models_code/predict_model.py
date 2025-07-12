# src/models_code/predict_model.py (DEFINITIVE v3 - Robust Inference)

import json
from pathlib import Path
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from PIL import Image
import pytesseract # <-- We need to run our own OCR
import pandas as pd
from io import StringIO

# --- Configuration ---
MODEL_PATH = "D:/invoice-parser-mlops/models/layoutlmv2-finetuned-sroie"
# Use a known file from the SROIE dataset to ensure a fair test.
TEST_IMAGE_PATH = "D:/invoice-parser-mlops/data/dataset/img/050.jpg" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load the EXACT model and processor from training ---
model = LayoutLMv2ForTokenClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
processor = LayoutLMv2Processor.from_pretrained(MODEL_PATH) # This processor has apply_ocr=False
id2label = model.config.id2label

# --- Helper Functions ---
def run_ocr(image):
    """Runs Tesseract OCR and returns words and boxes."""
    ocr_output = pytesseract.image_to_data(image, output_type=pytesseract.Output.STRING)
    df = pd.read_csv(StringIO(ocr_output), sep='\t', quoting=3)
    df = df.dropna()
    df = df[df.conf != -1]
    
    words = df['text'].tolist()
    boxes = []
    for i in range(len(df)):
        row = df.iloc[i]
        x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
        boxes.append([x, y, x + w, y + h])
        
    return words, boxes

def normalize_box(box, width, height):
    """Normalizes a bounding box to a [0, 1000] range."""
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]

# In predict_model.py

def smart_join(tokens):
    """
    Joins a list of tokens into a single string, handling sub-words
    and punctuation correctly.
    """
    if not tokens:
        return ""
    
    full_string = tokens[0]
    for token in tokens[1:]:
        # If it's a sub-word, attach it directly
        if token.startswith("##"):
            full_string += token[2:]
        # If it's punctuation, attach it directly (with some exceptions)
        elif token in ('.', ',', '!', '?', ':', ';', ')'):
            full_string += token
        # If the previous character was a space or it's certain punctuation, attach directly
        elif full_string.endswith(('(', '$', '€', '£', '¥')):
            full_string += token
        # Otherwise, add a space
        else:
            full_string += " " + token
            
    return full_string.strip()


def group_and_clean_entities(predictions, tokens):
    """
    Groups B- and I- tags using a more robust method and smart joining.
    """
    entities = []
    current_entity_tokens = []
    current_entity_tag = None

    for i, token_str in enumerate(tokens):
        if token_str in (processor.tokenizer.cls_token, processor.tokenizer.sep_token, processor.tokenizer.pad_token):
            continue
        
        pred_label = id2label[predictions[i]]
        
        if pred_label.startswith("B-"):
            if current_entity_tag:
                entities.append({"tag": current_entity_tag, "tokens": current_entity_tokens})
            
            current_entity_tag = pred_label[2:]
            current_entity_tokens = [token_str]

        elif pred_label.startswith("I-"):
            tag = pred_label[2:]
            if tag == current_entity_tag:
                current_entity_tokens.append(token_str)
            else:
                if current_entity_tag:
                    entities.append({"tag": current_entity_tag, "tokens": current_entity_tokens})
                current_entity_tag = tag
                current_entity_tokens = [token_str]
        
        else: # O-tag
            if current_entity_tag:
                entities.append({"tag": current_entity_tag, "tokens": current_entity_tokens})
            current_entity_tag = None
            current_entity_tokens = []
            
    if current_entity_tag:
        entities.append({"tag": current_entity_tag, "tokens": current_entity_tokens})
        
    # Now, process the collected tokens for each entity using our smart_join function
    final_results = {}
    for entity in entities:
        tag = entity['tag'].lower()
        text = smart_join(entity['tokens'])
        
        if tag in final_results:
            final_results[tag] += " " + text
        else:
            final_results[tag] = text
            
    return final_results

# --- Main Prediction Function ---
def predict_invoice(image_path: str):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 1. Run our own OCR, replicating the training data pipeline
    words, boxes = run_ocr(image)

    # 2. Normalize the bounding boxes
    normalized_boxes = [normalize_box(box, width, height) for box in boxes]

    # 3. Use the loaded processor (with apply_ocr=False)
    encoding = processor(
        image, words, boxes=normalized_boxes, return_tensors="pt",
        padding="max_length", truncation=True
    )
    
    # Move all tensors to the correct device
    for key, tensor in encoding.items():
        encoding[key] = tensor.to(DEVICE)
    
    # 4. Predict
    with torch.no_grad():
        outputs = model(**encoding)

    # 5. Post-process
    token_predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"].squeeze().tolist())
    
    # *** CRUCIAL DEBUGGING STEP ***
    print("\n--- Raw Predictions (Token by Token) ---")
    for token, pred_id in zip(tokens, token_predictions):
        if token not in (processor.tokenizer.cls_token, processor.tokenizer.sep_token, processor.tokenizer.pad_token):
            label = id2label[pred_id]
            if label != "O":
                print(f"{token:<20} -> {label}")
    # ********************************
    
    results = group_and_clean_entities(token_predictions, tokens)
    return results

# --- Script Execution ---
if __name__ == '__main__':
    print(f"--- Running Prediction on: {TEST_IMAGE_PATH} ---")
    try:
        extracted_data = predict_invoice(TEST_IMAGE_PATH)
        print("\n--- Extracted Data ---")
        if not extracted_data:
            print("Model did not extract any entities after grouping.")
        else:
            print(json.dumps(extracted_data, indent=2))
    except FileNotFoundError:
        print(f"\nERROR: Test image not found at '{TEST_IMAGE_PATH}'")