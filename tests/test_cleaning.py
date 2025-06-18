"""
Basic tests for the EDA app core functionality.
"""
import unittest
import pandas as pd
import numpy as np
from core.cleaning import detect_non_standard_missing, normalize_column

class TestCleaning(unittest.TestCase):
    def setUp(self):
        """Set up test data."""
        # Create a simple test dataframe
        self.df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'with_missing': [1, 'N/A', 3, '-', 5],
            'categorical': ['A', 'B', 'A', 'C', 'B']
        })
    
    def test_detect_non_standard_missing(self):
        """Test that non-standard missing values are correctly detected."""
        missing_dict = detect_non_standard_missing(self.df)
        
        # Should detect 'N/A' and '-' in 'with_missing' column
        self.assertIn('with_missing', missing_dict)
        self.assertEqual(len(missing_dict['with_missing']), 2)
        self.assertIn('n/a', [str(val).lower() for val in missing_dict['with_missing']])
        self.assertIn('-', [str(val).lower() for val in missing_dict['with_missing']])
    
    def test_normalize_column(self):
        """Test that column normalization works."""
        # Test min-max scaling
        normalized = normalize_column(self.df['numeric'], method='minmax')
        self.assertEqual(normalized.min(), 0.0)
        self.assertEqual(normalized.max(), 1.0)
        
        # Test z-score standardization
        normalized = normalize_column(self.df['numeric'], method='zscore')
        self.assertAlmostEqual(normalized.mean(), 0.0, places=10)
        self.assertAlmostEqual(normalized.std(), 1.0, places=10)
        
        # Test that categorical columns are returned unchanged
        normalized = normalize_column(self.df['categorical'])
        self.assertTrue(all(normalized == self.df['categorical']))

if __name__ == '__main__':
    unittest.main()
