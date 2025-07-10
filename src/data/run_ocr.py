import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from PIL import Image
import pytesseract
from tqdm import tqdm

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- On Windows, you might need to set the Tesseract command path explicitly ---
# Uncomment and update the path if Tesseract is not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "dataset"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"

# Input file from the previous step
SPLIT_IDS_FILE = INTERIM_DATA_DIR / "split_ids.json"
# Output directory for this step's results
OCR_OUTPUT_DIR = INTERIM_DATA_DIR / "ocr_outputs"

def get_all_ids(split_file: Path) -> List[str]:
    """Loads the split IDs JSON and returns a single list of all unique IDs."""
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    all_ids = split_data['train'] + split_data['val'] + split_data['test']
    logging.info(f"Loaded {len(all_ids)} total document IDs to process.")
    return sorted(list(set(all_ids)))

def process_single_image(file_id: str) -> List[Dict[str, Any]]:
    """
    Runs OCR on a single image and returns a structured list of words and their coordinates.
    """
    image_path = RAW_DATA_DIR / "img" / f"{file_id}.jpg" # Assumes .jpg, adjust if needed
    if not image_path.exists():
        logging.warning(f"Image file not found for ID {file_id}. Skipping.")
        return None
    
    try:
        image = Image.open(image_path)
        # Use image_to_data to get detailed information including bounding boxes
        ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        processed_words = []
        n_boxes = len(ocr_data['level'])
        for i in range(n_boxes):
            # We only care about actual words (level 5)
            if ocr_data['level'][i] == 5:
                text = ocr_data['text'][i].strip()
                if text: # Ensure we don't save empty strings
                    processed_words.append({
                        "text": text,
                        "left": ocr_data['left'][i],
                        "top": ocr_data['top'][i],
                        "width": ocr_data['width'][i],
                        "height": ocr_data['height'][i],
                        "conf": int(ocr_data['conf'][i])
                    })
        return processed_words

    except Exception as e:
        logging.error(f"Error processing image {file_id}: {e}")
        return None

def main():
    """
    Main function to run the OCR process for all documents.
    """
    if not SPLIT_IDS_FILE.exists():
        logging.error(f"Split file not found at {SPLIT_IDS_FILE}. Please run create_splits.py first.")
        return
        
    # Ensure the output directory exists
    OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_ids = get_all_ids(SPLIT_IDS_FILE)
    
    logging.info(f"Starting OCR process for {len(all_ids)} documents...")
    
    for file_id in tqdm(all_ids, desc="Running OCR"):
        output_path = OCR_OUTPUT_DIR / f"{file_id}.json"
        
        # Skip if already processed to allow for resuming
        if output_path.exists():
            continue
            
        result = process_single_image(file_id)
        if result is not None:
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
    
    logging.info("OCR processing complete.")
    logging.info(f"Results saved in: {OCR_OUTPUT_DIR}")

if __name__ == '__main__':
    main()