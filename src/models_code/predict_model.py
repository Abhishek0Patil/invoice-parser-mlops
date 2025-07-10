# src/models_code/predict_model.py (DEFINITIVE v2)

import json
from pathlib import Path
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from PIL import Image

# --- Configuration ---
MODEL_PATH = "D:/invoice-parser-mlops/models/layoutlmv2-finetuned-sroie"
# Use a known file from the SROIE dataset to ensure a fair test.
TEST_IMAGE_PATH = "D:/invoice-parser-mlops/data/dataset/img/005.jpg" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Model and Processor ---
# Best Practice: Load both from the same local directory.
model = LayoutLMv2ForTokenClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
processor = LayoutLMv2Processor.from_pretrained(MODEL_PATH)
id2label = model.config.id2label

# --- Helper Functions ---
def normalize_box(box, width, height):
    return [
        min(999, int(1000 * (box[0] / width))),
        min(999, int(1000 * (box[1] / height))),
        min(999, int(1000 * (box[2] / width))),
        min(999, int(1000 * (box[3] / height))),
    ]

def group_and_clean_entities(predictions, tokens):
    entities, current_entity = [], None
    for i, token_str in enumerate(tokens):
        if token_str in ("[CLS]", "[SEP]", "[PAD]"): continue
        pred_label = id2label[predictions[i]]
        if pred_label.startswith("B-"):
            if current_entity: entities.append(current_entity)
            current_entity = {"tag": pred_label[2:], "text": token_str}
        elif pred_label.startswith("I-") and current_entity and pred_label[2:] == current_entity["tag"]:
            current_entity["text"] += token_str[2:] if token_str.startswith("##") else " " + token_str
        else:
            if current_entity: entities.append(current_entity)
            current_entity = None
    if current_entity: entities.append(current_entity)
    final_results = {}
    for entity in entities:
        tag, text = entity['tag'].lower(), entity['text']
        final_results[tag] = (final_results.get(tag, "") + text + " ").strip()
    return final_results

# --- Main Prediction Function ---
def predict_invoice(image_path: str):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    # 1. Let the processor run OCR and prepare all inputs.
    # The processor loaded from the local directory already has apply_ocr=True if saved that way.
    # If not, we can re-initialize it. For safety, let's re-initialize.
    inference_processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=True)
    encoding = inference_processor(image, return_tensors="pt")
    
    # 2. Create the model_inputs dictionary
    model_inputs = {
        'image': encoding['image'].to(DEVICE),
        'input_ids': encoding['input_ids'].to(DEVICE),
        'attention_mask': encoding['attention_mask'].to(DEVICE),
        'token_type_ids': encoding['token_type_ids'].to(DEVICE),
    }

    # 3. Normalize the bounding boxes separately and add to the dictionary
    pixel_boxes = encoding['bbox'][0].tolist()
    normalized_boxes = [normalize_box(box, width, height) for box in pixel_boxes]
    model_inputs['bbox'] = torch.tensor([normalized_boxes]).to(DEVICE)

    # 4. Predict
    with torch.no_grad():
        outputs = model(**model_inputs)

    # 5. Post-process
    token_predictions = outputs.logits.argmax(-1).squeeze().tolist()
    tokens = inference_processor.tokenizer.convert_ids_to_tokens(model_inputs["input_ids"].squeeze().tolist())
    
    results = group_and_clean_entities(token_predictions, tokens)
    return results

# --- Script Execution ---
if __name__ == '__main__':
    print(f"--- Running Prediction on: {TEST_IMAGE_PATH} ---")
    try:
        extracted_data = predict_invoice(TEST_IMAGE_PATH)
        print("\n--- Extracted Data ---")
        if not extracted_data:
            print("Model did not extract any entities.")
        else:
            print(json.dumps(extracted_data, indent=2))
    except FileNotFoundError:
        print(f"\nERROR: Test image not found at '{TEST_IMAGE_PATH}'")