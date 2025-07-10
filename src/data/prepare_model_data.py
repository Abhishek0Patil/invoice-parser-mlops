import json
import logging
from pathlib import Path
import re
from datasets import Dataset, DatasetDict, Features, ClassLabel, Value, Sequence

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "dataset"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

SPLIT_IDS_FILE = INTERIM_DATA_DIR / "split_ids.json"
OCR_OUTPUT_DIR = INTERIM_DATA_DIR / "ocr_outputs"

# --- Labeling Configuration ---
# Define the entities you care about from your ground-truth JSON files
LABEL_KEYS = ["company", "date", "address", "total"]
# Create the list of B-I-O labels
LABELS = ["O"] + [f"B-{key.upper()}" for key in LABEL_KEYS] + [f"I-{key.upper()}" for key in LABEL_KEYS]

def normalize_text(text: str) -> str:
    """A simple normalization function to improve matching."""
    return re.sub(r'[\W_]+', '', text).lower()

def create_dataset_from_splits(split_ids: dict):
    """
    The main function to process all documents and create the final Hugging Face Dataset.
    """
    all_data = { "train": [], "validation": [], "test": [] }
    
    # Map splits from file (val) to dataset convention (validation)
    split_map = {'train': 'train', 'val': 'validation', 'test': 'test'}

    for split_name_file, split_name_dataset in split_map.items():
        logging.info(f"Processing split: {split_name_dataset}")
        
        for file_id in split_ids[split_name_file]:
            ground_truth_path = RAW_DATA_DIR / "label" / f"{file_id}.json"
            ocr_path = OCR_OUTPUT_DIR / f"{file_id}.json"
            
            if not ground_truth_path.exists() or not ocr_path.exists():
                logging.warning(f"Skipping {file_id}: missing ground truth or ocr file.")
                continue

            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            with open(ocr_path, 'r', encoding='utf-8') as f:
                ocr_data = json.load(f)

            # Filter out low-confidence OCR words
            words = [word['text'] for word in ocr_data if word['conf'] > 30]
            boxes = [[word['left'], word['top'], word['left'] + word['width'], word['top'] + word['height']] 
                     for word in ocr_data if word['conf'] > 30]
            
            # Initialize all labels to 'O' (Outside)
            word_labels = ["O"] * len(words)

            # --- The Alignment Logic ---
            for key, value in ground_truth.items():
                if not value or key not in LABEL_KEYS:
                    continue
                
                # Normalize and split the ground-truth value into words
                target_words = [normalize_text(w) for w in str(value).split() if normalize_text(w)]
                if not target_words: continue

                # Search for the sequence of target words in the OCR'd words
                for i in range(len(words) - len(target_words) + 1):
                    match = True
                    for j in range(len(target_words)):
                        ocr_word = normalize_text(words[i+j])
                        if target_words[j] not in ocr_word:
                            match = False
                            break
                    
                    if match:
                        # Found a match, apply B-I-O tags
                        word_labels[i] = f"B-{key.upper()}"
                        for k in range(1, len(target_words)):
                            word_labels[i+k] = f"I-{key.upper()}"
                        break # Move to the next key-value pair once a match is found
            
            all_data[split_name_dataset].append({
                "id": file_id,
                "words": words,
                "bboxes": boxes,
                "ner_tags": [LABELS.index(lbl) for lbl in word_labels]
            })

    # --- Create the Hugging Face Dataset ---
    features = Features({
        'id': Value('string'),
        'words': Sequence(Value('string')),
        'bboxes': Sequence(Sequence(Value('int64'))),
        'ner_tags': Sequence(ClassLabel(names=LABELS))
    })

    train_dataset = Dataset.from_list(all_data['train'], features=features)
    val_dataset = Dataset.from_list(all_data['validation'], features=features)
    test_dataset = Dataset.from_list(all_data['test'], features=features)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


def main():
    """Main execution function"""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(SPLIT_IDS_FILE, 'r') as f:
        split_ids = json.load(f)
        
    final_dataset = create_dataset_from_splits(split_ids)
    
    logging.info(f"Final dataset created:\n{final_dataset}")
    
    # Save the dataset to disk
    output_path = PROCESSED_DATA_DIR / "sroie_dataset_for_layoutlm"
    final_dataset.save_to_disk(output_path)
    logging.info(f"Dataset successfully saved to {output_path}")

if __name__ == '__main__':
    main()