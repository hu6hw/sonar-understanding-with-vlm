import os
from pathlib import Path
import pandas as pd
import ollama
from datetime import datetime

# --- Config ---
DATASET_PATH = 'md_fls_dataset/data/watertank-cropped'

PROMPT = 'What is the following image? ONLY choose from the following classes, NO other response is allowed, and your response MUST be only one word; bottle, can, chain, drink-carton, hook, propeller, shampoo-bottle, standing-bottle, tire, valve.'
MODEL = 'llama3.2-vision:11b'

VALID_CLASSES = {
    'can', 'bottle', 'drink-carton', 'chain', 'propeller',
    'tire', 'hook', 'valve', 'shampoo-bottle', 'standing-bottle'
}

IMAGE_EXTS = {'.png'}

BREAK_AFTER = 1 # Stop running inference after n images. If -1 then don't break

# Resolve paths
dataset_root = Path(DATASET_PATH)

chosen_classes = VALID_CLASSES
for class_name in chosen_classes:
    subset_dir = dataset_root / class_name

    # If subset folder not found then raise error
    if not subset_dir.exists() or not subset_dir.is_dir():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")

    # Get all image paths in the subset folder recursively
    image_paths = sorted(
        p for p in subset_dir.rglob('*') if p.suffix.lower() in IMAGE_EXTS and p.is_file()
    )

    # If no images found then raise error
    if not image_paths:
        raise FileNotFoundError(f"No images found in subset directory: {subset_dir}")

    records = []

    print(datetime.now().astimezone().strftime("Started at: %Y-%m-%d %H:%M:%S %Z"))
    print(f"Found {len(image_paths)} images in '{subset_dir}'. Running inference with model '{MODEL}'...")
    if BREAK_AFTER > -1: print(f"Stop after {BREAK_AFTER} image(s)")

    for i, img in enumerate(image_paths, start=1):
        try:
            resp = ollama.generate(model=MODEL, prompt=PROMPT, images=[str(img)])

            # Handle different response types properly
            if isinstance(resp, dict):
                raw_text = resp.get('response', '')
            else:
                # Some versions return an object with a .response attr
                raw_text = getattr(resp, 'response', '')

            text = (raw_text or '').strip()

            # Normalize prediction
            normalized = text.lower().strip().strip('.').strip()

            predicted_class = normalized
            records.append({
                'image_path': str(img),
                'actual_class': class_name,
                'predicted_class': predicted_class,
                'raw_response': text,
                'model': MODEL
            })

            # Progress logging
            if i % 10 == 0 or i == len(image_paths):
                print(f"Processed {i}/{len(image_paths)} images...")
                
            if BREAK_AFTER > -1 and i == BREAK_AFTER: 
                break

        except Exception as e:
            # Record the error and continue
            records.append({
                'image_path': str(img),
                'actual_class': class_name,
                'predicted_class': None,
                'raw_response': f'ERROR: {e}',
                'model': MODEL
            })
            print(f"Error processing {img}: {e}")

    # Create DataFrame and save to CSV
    df = pd.DataFrame.from_records(records, columns=[
        'image_path', 'actual_class', 'predicted_class', 'raw_response', 'model'
    ])
    csv_name = f"{class_name}_{MODEL.replace(':', '-')}_predictions.csv"
    csv_path = Path.cwd() / csv_name
    df.to_csv(csv_path, index=False)

    print(f"Done. Saved results to: {csv_path}")
    print(datetime.now().astimezone().strftime("Finished at: %Y-%m-%d %H:%M:%S %Z"))
