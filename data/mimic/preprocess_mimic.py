"""
MIMIC-CXR Dataset Preprocessing Script

This script processes the MIMIC-CXR dataset for fairness research.
It performs the following:
1. Loads metadata and encodes sensitive attributes (sex, race)
2. Splits data into train/val/test sets by study_id (no patient overlap)
3. Saves processed metadata CSVs

Prerequisites:
- Download MIMIC-CXR dataset from PhysioNet (requires credentialing)
- Place metadata.csv in the same directory as this script
- Images should be in the appropriate folder structure

Usage:
    python preprocess_mimic.py --input_path ./metadata.csv --output_path ./
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import argparse
import os


def process_metadata(input_path, output_path, seed=42):
    """
    Process MIMIC-CXR metadata file.
    
    Args:
        input_path: Path to the raw metadata.csv
        output_path: Directory to save processed files
        seed: Random seed for reproducibility
    """
    print(f"Loading metadata from {input_path}...")
    data = pd.read_csv(input_path)
    
    # Encode sex: M=1, F=0
    data['sex'] = data['sex'].replace({'M': 1, 'F': 0})
    
    # Filter out 'OTHER' race and encode remaining
    data = data[data['race'] != 'OTHER']
    race_mapping = {'WHITE': 0, 'BLACK': 1, 'ASIAN': 2, 'HISPANIC/LATINO': 3}
    data['race'] = data['race'].replace(race_mapping)
    
    # Create age groups
    data['Age_multi'] = data['age_ori'].values.astype('int')
    data['Age_multi'] = np.where(data['Age_multi'].between(-1, 19), 0, data['Age_multi'])
    data['Age_multi'] = np.where(data['Age_multi'].between(20, 39), 1, data['Age_multi'])
    data['Age_multi'] = np.where(data['Age_multi'].between(40, 59), 2, data['Age_multi'])
    data['Age_multi'] = np.where(data['Age_multi'].between(60, 79), 3, data['Age_multi'])
    data['Age_multi'] = np.where(data['Age_multi'] >= 80, 4, data['Age_multi'])
    
    print(f"Total samples after filtering: {len(data)}")
    print(f"Race distribution:\n{data['race'].value_counts()}")
    print(f"Sex distribution:\n{data['sex'].value_counts()}")
    
    # Save intermediate file
    intermediate_path = os.path.join(output_path, 'meta_onehot.csv')
    data.to_csv(intermediate_path, index=False)
    print(f"Saved intermediate file to {intermediate_path}")
    
    return data


def split_by_study_id(data, output_path, seed=42):
    """
    Split data by study_id to prevent patient overlap between splits.
    
    Args:
        data: Processed DataFrame
        output_path: Directory to save split files
        seed: Random seed for reproducibility
    """
    # Get unique study_ids
    study_ids = data['study_id'].unique()
    print(f"Total unique study IDs: {len(study_ids)}")
    
    # Split study_ids: 80% train, 10% val, 10% test
    train_ids, test_val_ids = train_test_split(study_ids, test_size=0.2, random_state=seed)
    test_ids, val_ids = train_test_split(test_val_ids, test_size=0.5, random_state=seed)
    
    # Map splits back to data
    train_data = data[data['study_id'].isin(train_ids)]
    val_data = data[data['study_id'].isin(val_ids)]
    test_data = data[data['study_id'].isin(test_ids)]
    
    # Verify no overlap
    train_set = set(train_data['study_id'])
    val_set = set(val_data['study_id'])
    test_set = set(test_data['study_id'])
    
    assert len(train_set & val_set) == 0, "Train and val have overlapping study_ids!"
    assert len(train_set & test_set) == 0, "Train and test have overlapping study_ids!"
    assert len(val_set & test_set) == 0, "Val and test have overlapping study_ids!"
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_data)} samples ({len(train_ids)} study_ids)")
    print(f"  Val: {len(val_data)} samples ({len(val_ids)} study_ids)")
    print(f"  Test: {len(test_data)} samples ({len(test_ids)} study_ids)")
    
    # Save split files
    train_data.to_csv(os.path.join(output_path, 'metadata_train_number.csv'), index=False)
    val_data.to_csv(os.path.join(output_path, 'metadata_val_number.csv'), index=False)
    test_data.to_csv(os.path.join(output_path, 'metadata_test_number.csv'), index=False)
    
    print(f"\nSaved split files to {output_path}")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(description='Preprocess MIMIC-CXR dataset')
    parser.add_argument('--input_path', type=str, default='./metadata.csv',
                        help='Path to raw metadata CSV, we used 50,000 samples from the original dataset')
    parser.add_argument('--output_path', type=str, default='./',
                        help='Directory to save processed files')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    
    # Process metadata
    data = process_metadata(args.input_path, args.output_path, args.seed)
    
    # Split data
    split_by_study_id(data, args.output_path, args.seed)
    
    print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

