import json
import logging
from pathlib import Path
from typing import List
import random

from sklearn.model_selection import train_test_split

# --- Configuration ---
# Configure logging for clear, informative output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use pathlib for robust, cross-platform path handling
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
SOURCE_DATA_DIR = ROOT_DIR / "data" / "dataset"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"

# The final JSON file where the split IDs will be saved
SPLIT_IDS_FILE = INTERIM_DATA_DIR / "split_ids.json"

# --- Parameters for the Split ---
TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
# Using a fixed random state ensures that the split is the same every time you run it.
# This is crucial for reproducibility!
RANDOM_STATE = 42

def gather_valid_file_ids(source_dir: Path) -> List[str]:
    """
    Scans a single data source directory to find all documents that have
    both an image and a corresponding JSON label.

    Args:
        source_dir: The path to the directory containing 'img' and 'label' subfolders.

    Returns:
        A sorted list of unique file identifiers (stems).
    """
    logging.info(f"Scanning for valid documents in '{source_dir}'...")
    image_dir = source_dir / "img"
    label_dir = source_dir / "label"
    
    if not image_dir.is_dir() or not label_dir.is_dir():
        logging.error(f"Required 'img' and/or 'label' folders not found in '{source_dir}'.")
        return []
        
    valid_ids = []
    # Iterate through all files in the image directory
    for image_path in image_dir.glob('*'):
        # The 'stem' is the filename without the extension (e.g., 'X51005200619')
        file_id = image_path.stem
        
        # CRITICAL CHECK: Ensure a corresponding label file exists.
        # This prevents including images without ground truth in our dataset.
        label_path = label_dir / f"{file_id}.json"
        
        if label_path.exists():
            valid_ids.append(file_id)
        else:
            logging.warning(f"Found image '{image_path.name}' but no corresponding label. Skipping.")
    
    # Sorting ensures a consistent order, though shuffling will happen later
    unique_ids = sorted(list(set(valid_ids)))
    logging.info(f"Total unique and valid document IDs found: {len(unique_ids)}")
    return unique_ids

def create_and_save_splits(file_ids: List[str]):
    """
    Splits the list of file IDs into train, validation, and test sets and saves them.
    This function's logic is identical to the previous version, as it's perfectly
    suited for this task.
    """
    if not file_ids:
        logging.error("No file IDs provided. Cannot create splits.")
        return

    # Ensure the split ratios sum to 1.0
    if not abs((TRAIN_SIZE + VAL_SIZE + TEST_SIZE) - 1.0) < 1e-9:
        raise ValueError("Sum of TRAIN_SIZE, VAL_SIZE, and TEST_SIZE must be 1.0")

    # First split: Separate the training set from the combined validation + test set
    train_ids, temp_ids = train_test_split(
        file_ids,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Calculate the proportion of the test set relative to the temporary set's size
    relative_test_size = TEST_SIZE / (VAL_SIZE + TEST_SIZE)
    
    # Second split: Split the temporary set into validation and test sets
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    logging.info("--- Split Summary ---")
    logging.info(f"Total documents:     {len(file_ids)}")
    logging.info(f"Training set size:   {len(train_ids)} ({len(train_ids)/len(file_ids):.2%})")
    logging.info(f"Validation set size: {len(val_ids)} ({len(val_ids)/len(file_ids):.2%})")
    logging.info(f"Test set size:       {len(test_ids)} ({len(test_ids)/len(file_ids):.2%})")

    # Prepare data for saving to JSON
    split_data = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }

    # Ensure the output directory exists before writing the file
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with open(SPLIT_IDS_FILE, 'w') as f:
        json.dump(split_data, f, indent=4)

    logging.info(f"Successfully saved split IDs to '{SPLIT_IDS_FILE}'")


if __name__ == '__main__':
    # Set a seed for Python's built-in random module for an extra layer of reproducibility
    random.seed(RANDOM_STATE)
    
    all_file_ids = gather_valid_file_ids(SOURCE_DATA_DIR)
    create_and_save_splits(all_file_ids)