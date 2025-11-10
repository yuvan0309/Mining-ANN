#!/usr/bin/env python3
"""
Data ingestion module for FoS prediction with Ru (pore pressure ratio) incorporation.
Based on Bishop's Simplified Method with pore pressure.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_prepare_data(csv_path, include_ru=True):
    """
    Load data from CSV and prepare features including Ru values.
    
    Parameters:
    -----------
    csv_path : str or Path
        Path to the Overall Data.csv file
    include_ru : bool
        Whether to include Ru values as features (default: True)
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix with cohesion, friction angle, unit weight, and optionally Ru
    y : pandas.Series
        Target variable (FoS values)
    """
    # Read the CSV file - no headers
    df = pd.read_csv(csv_path, header=None)
    
    # Extract data rows - skip headers
    # The CSV has multiple sections, we need to parse carefully
    data_rows = []
    
    # Parse the CSV structure
    for idx, row in df.iterrows():
        # Skip empty rows or header rows
        if pd.isna(row.iloc[1]) or str(row.iloc[1]).strip() in ['', 'Material', 'Point 1', 'Point 2', 'Point 3', 
                                                                   'Point 4', 'Point 5', 'Point 6', 'Point 7', 
                                                                   'Point 8', 'Point 9', 'Point 10', 'Cohesion (kPa)']:
            continue
        
        # Look for material rows with numerical cohesion value
        try:
            material_name = str(row.iloc[1]).strip()
            
            # Try to convert to float - if it's a material name, this will fail
            try:
                float(material_name)
                continue  # Skip if it's a number (not a material name)
            except (ValueError, TypeError):
                pass  # It's likely a material name
            
            # Pre-monsoon without Ru (columns 1-6)
            if not pd.isna(row.iloc[5]) and row.iloc[5] != '':  # FoS value exists
                try:
                    cohesion = float(row.iloc[2])
                    friction_angle = float(row.iloc[3])
                    unit_weight = float(row.iloc[4])
                    fos = float(row.iloc[5])
                    ru = float(row.iloc[6]) if not pd.isna(row.iloc[6]) and str(row.iloc[6]).strip() != '' else 0.0
                    
                    data_rows.append({
                        'material': material_name,
                        'cohesion': cohesion,
                        'friction_angle': friction_angle,
                        'unit_weight': unit_weight,
                        'fos': fos,
                        'ru': ru,
                        'season': 'pre_monsoon',
                        'ru_applied': False
                    })
                except (ValueError, TypeError):
                    pass
            
            # Pre-monsoon with Ru (columns 8-13)
            if len(row) > 12 and not pd.isna(row.iloc[12]):
                try:
                    cohesion = float(row.iloc[9])
                    friction_angle = float(row.iloc[10])
                    unit_weight = float(row.iloc[11])
                    fos = float(row.iloc[12])
                    ru = float(row.iloc[13]) if len(row) > 13 and not pd.isna(row.iloc[13]) else 0.0
                    
                    data_rows.append({
                        'material': material_name,
                        'cohesion': cohesion,
                        'friction_angle': friction_angle,
                        'unit_weight': unit_weight,
                        'fos': fos,
                        'ru': ru,
                        'season': 'pre_monsoon',
                        'ru_applied': True
                    })
                except (ValueError, TypeError):
                    pass
            
            # Post-monsoon without Ru (columns 15-20)
            if len(row) > 19 and not pd.isna(row.iloc[19]):
                try:
                    cohesion = float(row.iloc[16])
                    friction_angle = float(row.iloc[17])
                    unit_weight = float(row.iloc[18])
                    fos = float(row.iloc[19])
                    ru = float(row.iloc[20]) if len(row) > 20 and not pd.isna(row.iloc[20]) else 0.0
                    
                    data_rows.append({
                        'material': material_name,
                        'cohesion': cohesion,
                        'friction_angle': friction_angle,
                        'unit_weight': unit_weight,
                        'fos': fos,
                        'ru': ru,
                        'season': 'post_monsoon',
                        'ru_applied': False
                    })
                except (ValueError, TypeError):
                    pass
            
            # Post-monsoon with Ru (columns 22-27)
            if len(row) > 26 and not pd.isna(row.iloc[26]):
                try:
                    cohesion = float(row.iloc[23])
                    friction_angle = float(row.iloc[24])
                    unit_weight = float(row.iloc[25])
                    fos = float(row.iloc[26])
                    ru = float(row.iloc[27]) if len(row) > 27 and not pd.isna(row.iloc[27]) else 0.0
                    
                    data_rows.append({
                        'material': material_name,
                        'cohesion': cohesion,
                        'friction_angle': friction_angle,
                        'unit_weight': unit_weight,
                        'fos': fos,
                        'ru': ru,
                        'season': 'post_monsoon',
                        'ru_applied': True
                    })
                except (ValueError, TypeError):
                    pass
                    
        except (ValueError, IndexError, TypeError):
            continue
    
    # Create DataFrame
    data_df = pd.DataFrame(data_rows)
    
    # Remove duplicates
    data_df = data_df.drop_duplicates()
    
    # Remove rows where FoS <= 0 (invalid)
    data_df = data_df[data_df['fos'] > 0]
    
    # Remove rows with any NaN values
    data_df = data_df.dropna()
    
    # Reset index
    data_df = data_df.reset_index(drop=True)
    
    # Prepare features and target
    if include_ru:
        X = data_df[['cohesion', 'friction_angle', 'unit_weight', 'ru']]
    else:
        X = data_df[['cohesion', 'friction_angle', 'unit_weight']]
    
    y = data_df['fos']
    
    print(f"✓ Loaded {len(data_df)} samples")
    print(f"✓ Features: {list(X.columns)}")
    print(f"✓ FoS range: {y.min():.3f} - {y.max():.3f}")
    if include_ru:
        print(f"✓ Ru range: {data_df['ru'].min():.3f} - {data_df['ru'].max():.3f}")
    
    return X, y, data_df


def train_test_split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    test_size : float
        Proportion of test data (default: 0.2 = 20%)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"✓ Training samples: {len(X_train)} ({(1-test_size)*100:.0f}%)")
    print(f"✓ Testing samples: {len(X_test)} ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Test the data loading
    csv_path = Path(__file__).parent / "data" / "Overall Data.csv"
    X, y, df = load_and_prepare_data(csv_path, include_ru=True)
    
    print(f"\n📋 Dataset Info:")
    print(X.describe())
    print(f"\n📋 Target (FoS) Info:")
    print(y.describe())
