import pytest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'core')))

from eda_basic import (
    get_data_types_summary,
    plot_numeric_distribution,
    plot_categorical_distribution,
    plot_datetime_distribution,
    get_descriptive_stats,
    plot_boxplot,
)

@pytest.fixture
def sample_df():
    # Sample DataFrame for testing
    return pd.DataFrame({
        'num': [1, 2, 3, 4, 5],
        'cat': ['a', 'b', 'a', 'b', 'c'],
        'bool': [True, False, True, False, True],
        'text': ['lorem ipsum'] * 5,
        'date': pd.date_range('2021-01-01', periods=5),
        'other': [None, None, None, None, None]
    })

def test_get_data_types_summary(sample_df): 
    # Test the data types summary function
    # Ensure it returns a dictionary with expected keys
    result = get_data_types_summary(sample_df)
    assert isinstance(result, dict)
    assert 'Numeric' in result
    assert 'Boolean' in result
    assert 'DateTime' in result

def test_plot_numeric_distribution(sample_df):
    # Test the numeric distribution plotting function
    # Ensure it returns a figure object
    fig = plot_numeric_distribution(sample_df, 'num')
    assert fig is not None
    assert hasattr(fig, 'to_dict')

def test_plot_categorical_distribution(sample_df):
    # Test the categorical distribution plotting function
    # Ensure it returns a figure object
    fig = plot_categorical_distribution(sample_df, 'cat')
    assert fig is not None
    assert hasattr(fig, 'to_dict')

def test_plot_datetime_distribution(sample_df):
    # Test the datetime distribution plotting function
    # Ensure it returns a figure object
    fig = plot_datetime_distribution(sample_df, 'date', freq='M')
    assert fig is not None
    assert hasattr(fig, 'to_dict')

def test_get_descriptive_stats_default(sample_df):
    # Test the descriptive statistics function with default parameters
    # Ensure it returns a DataFrame with expected columns
    result = get_descriptive_stats(sample_df)
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns

def test_get_descriptive_stats_subset(sample_df):
    # Test the descriptive statistics function with a subset of columns
    # Ensure it returns a DataFrame with expected columns
    result = get_descriptive_stats(sample_df, ['num'])
    assert isinstance(result, pd.DataFrame)
    assert 'mean' in result.columns
    assert result.index[0] == 'num'

def test_get_descriptive_stats_empty(sample_df):
    # Test the descriptive statistics function with an empty DataFrame
    result = get_descriptive_stats(sample_df, ['text'])  # non-numeric
    assert result.empty

def test_plot_boxplot_basic(sample_df):
    # Test the boxplot function with basic parameters
    # Ensure it returns a figure object
    fig = plot_boxplot(sample_df, 'num')
    assert fig is not None
    assert hasattr(fig, 'to_dict')

def test_plot_boxplot_grouped(sample_df):
    # Test the boxplot function with grouping by a categorical variable
    # Ensure it returns a figure object
    fig = plot_boxplot(sample_df, 'num', 'cat')
    assert fig is not None
    assert hasattr(fig, 'to_dict')