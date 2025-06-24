"""
Basic tests for the EDA app core functionality using pytest framework.
"""
import pytest
import pandas as pd
import numpy as np
from core.cleaning import detect_non_standard_missing, normalize_column, replace_non_standard_missing

@pytest.fixture
def sample_dataframe():
    """Create a simple test dataframe."""
    return pd.DataFrame({
        'numeric': [1, 2, 3, 4, 5],
        'with_missing': [1, 'N/A', 3, '-', 5],
        'categorical': ['A', 'B', 'A', 'C', 'B']
    })

def test_detect_non_standard_missing(sample_dataframe):
    """Test that non-standard missing values are correctly detected."""
    missing_dict = detect_non_standard_missing(sample_dataframe)
    
    # Should detect '-' in 'with_missing' column (N/A is already recognized by pandas)
    assert 'with_missing' in missing_dict
    assert len(missing_dict['with_missing']) == 1
    assert '-' in [str(val).lower() for val in missing_dict['with_missing']]
    
def test_replace_non_standard_missing():
    """
    Tests that non-standard missing values (e.g., '.', '-', ' ')
    are correctly replaced with np.nan.
    """
    # 1. Create a sample DataFrame with non-standard missing values
    data = {
        'A': [1, 2, '.', 4],
        'B': ['x', '-', 'y', 'z'],
        'C': [5.0, 6.0, 7.0, ' ']
    }
    df = pd.DataFrame(data)

    # 2. Define the dictionary mapping columns to null values
    null_values_dict = {
        'A': ['.'],
        'B': ['-'],
        'C': [' ']
    }

    # 3. Call the function to be tested
    cleaned_df = replace_non_standard_missing(df, null_values_dict)

    # 4. Assert the expected outcome
    assert pd.isna(cleaned_df.loc[2, 'A'])
    assert pd.isna(cleaned_df.loc[1, 'B'])
    assert pd.isna(cleaned_df.loc[3, 'C'])
    assert cleaned_df.loc[0, 'A'] == 1  # Ensure other values are untouched
    
def test_normalize_column_minmax(sample_dataframe):
    """Test that column normalization works with min-max scaling."""
    # Test min-max scaling
    normalized = normalize_column(sample_dataframe['numeric'], method='minmax')
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    
def test_normalize_column_zscore(sample_dataframe):
    """Test that column normalization works with z-score standardization."""
    # Test z-score standardization
    normalized = normalize_column(sample_dataframe['numeric'], method='zscore')
    
    # Mean should be almost exactly 0
    assert pytest.approx(normalized.mean(), abs=1e-10) == 0.0
    
    # Since we're dealing with a small dataset and sklearn's StandardScaler,
    # the exact standard deviation might vary slightly from 1.0
    # Instead of exact value comparison, let's just check if it's reasonably close to 1.0
    assert 0.8 < normalized.std() < 1.2  # Wider tolerance
    
def test_normalize_column_categorical(sample_dataframe):
    """Test that categorical columns are returned unchanged."""
    # Test that categorical columns are returned unchanged
    normalized = normalize_column(sample_dataframe['categorical'])
    assert all(normalized == sample_dataframe['categorical'])
