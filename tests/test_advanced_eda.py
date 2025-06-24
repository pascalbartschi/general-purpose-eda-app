"""
Test module for advanced EDA functions.
"""
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal, assert_series_equal
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Add the parent directory to sys.path to import the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.eda_advanced import (
    calculate_correlation_with_pvalues,
    perform_pca,
    perform_tsne,
    perform_umap_reduction,
    perform_kmeans,
    perform_dbscan,
    perform_ttest,
    perform_anova
)

# Create sample data for testing
@pytest.fixture
def sample_data():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    data = {
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.normal(0, 1, 100),
        'feature3': np.random.normal(0, 1, 100),
        'categorical': np.random.choice(['A', 'B', 'C'], 100)
    }
    # Create a correlation between feature1 and feature2
    data['feature2'] = data['feature1'] * 0.7 + np.random.normal(0, 0.5, 100)
    
    return pd.DataFrame(data)

def test_calculate_correlation_with_pvalues(sample_data):
    """Test correlation calculation with p-values."""
    # Test with Pearson method
    corr_matrix, p_values = calculate_correlation_with_pvalues(sample_data, method='pearson')
    
    # Verify output shapes
    assert corr_matrix.shape == (3, 3)  # Only numeric columns
    assert p_values.shape == (3, 3)
    
    # Verify correlation between feature1 and feature2 is strong
    assert abs(corr_matrix.loc['feature1', 'feature2']) > 0.6
    
    # Verify p-value between feature1 and feature2 is small (statistically significant)
    assert p_values.loc['feature1', 'feature2'] < 0.05
    
    # Test diagonal values
    assert np.allclose(np.diag(corr_matrix), 1.0)
    assert np.allclose(np.diag(p_values), 0.0)
    
    # Test with Spearman method
    corr_spearman, p_spearman = calculate_correlation_with_pvalues(sample_data, method='spearman')
    assert corr_spearman.shape == (3, 3)
    
    # Test with Kendall method
    corr_kendall, p_kendall = calculate_correlation_with_pvalues(sample_data, method='kendall')
    assert corr_kendall.shape == (3, 3)
    
    # Test with invalid method
    with pytest.raises(ValueError):
        calculate_correlation_with_pvalues(sample_data, method='invalid_method')

def test_perform_pca(sample_data):
    """Test PCA implementation."""
    # Test with default parameters
    pca_results, loadings, explained_var = perform_pca(sample_data)
    
    # Verify output shapes
    assert pca_results.shape == (100, 2)  # Default n_components=2
    assert loadings.shape == (3, 2)  # 3 numeric features, 2 components
    assert len(explained_var) == 2
    
    # Test with different number of components
    pca_results, loadings, explained_var = perform_pca(sample_data, n_components=3)
    assert pca_results.shape == (100, 3)
    assert loadings.shape == (3, 3)
    assert len(explained_var) == 3
    
    # Test with standardize=False
    pca_results, loadings, explained_var = perform_pca(sample_data, standardize=False)
    assert pca_results.shape == (100, 2)
    
    # Test with more components than features
    pca_results, loadings, explained_var = perform_pca(sample_data, n_components=10)
    assert pca_results.shape == (100, 3)  # Should be limited to min(n_components, n_features)
    assert loadings.shape == (3, 3)
    assert len(explained_var) == 3

def test_perform_tsne(sample_data):
    """Test t-SNE implementation."""
    # Test with default parameters
    tsne_results = perform_tsne(sample_data)
    
    # Verify output shape
    assert tsne_results.shape == (100, 2)
    assert list(tsne_results.columns) == ['TSNE1', 'TSNE2']
    
    # Test with custom parameters
    tsne_results = perform_tsne(sample_data, perplexity=10, learning_rate=100.0, n_iter=250)
    assert tsne_results.shape == (100, 2)
    
    # Test with perplexity larger than n_samples-1
    tsne_results = perform_tsne(sample_data, perplexity=200)
    assert tsne_results.shape == (100, 2)  # Should adjust perplexity to n_samples-1

def test_perform_umap_reduction(sample_data):
    """Test UMAP implementation."""
    try:
        import umap
        
        # Test with default parameters
        umap_results = perform_umap_reduction(sample_data)
        
        # Verify output shape
        assert umap_results.shape == (100, 2)
        assert list(umap_results.columns) == ['UMAP1', 'UMAP2']
        
        # Test with custom parameters
        umap_results = perform_umap_reduction(sample_data, n_neighbors=5, min_dist=0.01)
        assert umap_results.shape == (100, 2)
        
        # Test with n_neighbors larger than n_samples-1
        umap_results = perform_umap_reduction(sample_data, n_neighbors=200)
        assert umap_results.shape == (100, 2)  # Should adjust n_neighbors to n_samples-1
    except ImportError:
        pytest.skip("umap-learn not installed, skipping UMAP tests")

def test_perform_kmeans(sample_data):
    """Test KMeans clustering implementation."""
    # Test with default parameters
    clusters, kmeans_model = perform_kmeans(sample_data)
    
    # Verify output shape
    assert clusters.shape == (100, 1)
    assert 'Cluster' in clusters.columns
    assert len(np.unique(clusters['Cluster'])) <= 3  # Default n_clusters=3
    
    # Verify model is returned correctly
    assert isinstance(kmeans_model, KMeans)
    assert kmeans_model.n_clusters == 3
    
    # Test with custom parameters
    clusters, kmeans_model = perform_kmeans(sample_data, n_clusters=5)
    assert len(np.unique(clusters['Cluster'])) <= 5
    assert kmeans_model.n_clusters == 5

def test_perform_dbscan(sample_data):
    """Test DBSCAN clustering implementation."""
    # Test with default parameters
    clusters = perform_dbscan(sample_data)
    
    # Verify output shape
    assert clusters.shape == (100, 1)
    assert 'Cluster' in clusters.columns
    
    # Test with custom parameters
    clusters = perform_dbscan(sample_data, eps=0.2, min_samples=3)
    assert 'Cluster' in clusters.columns

def test_perform_ttest(sample_data):
    """Test t-test implementation."""
    # Test with valid data
    results = perform_ttest(sample_data, 'feature1', 'categorical')
    
    # Verify output
    assert isinstance(results, pd.DataFrame)
    if not results.empty:  # If enough data in groups
        expected_columns = [
            'Group 1', 'Group 2', 'Mean 1', 'Mean 2', 'Std Dev 1', 'Std Dev 2',
            't-statistic', 'p-value', "Cohen's d", 'Significant'
        ]
        assert all(col in results.columns for col in expected_columns)
    
    # Test with invalid column
    results = perform_ttest(sample_data, 'feature1', 'nonexistent_column')
    assert results.empty
    
    # Test with single value in a group
    small_df = pd.DataFrame({
        'value': [1, 2, 3, 4],
        'group': ['A', 'A', 'B', 'C']
    })
    results = perform_ttest(small_df, 'value', 'group')
    # Should skip groups with only 1 value
    assert len(results) <= 3  # Maximum 3 possible pairs (A-B, A-C, B-C)

def test_perform_anova(sample_data):
    """Test ANOVA implementation."""
    # Test with valid data
    results, group_stats = perform_anova(sample_data, 'feature1', 'categorical')
    
    # Verify output
    assert isinstance(results, dict)
    assert isinstance(group_stats, pd.DataFrame)
    
    expected_keys = [
        'f_statistic', 'p_value', 'eta_squared', 'significant', 
        'n_groups', 'n_samples'
    ]
    assert all(key in results for key in expected_keys)
    
    expected_columns = ['Group', 'Count', 'Mean', 'Std Dev', 'Min', 'Max']
    assert all(col in group_stats.columns for col in expected_columns)
    
    # Test with insufficient groups
    small_df = pd.DataFrame({
        'value': [1, 2, 3],
        'group': ['A', 'A', 'A']  # Only one group
    })
    results, group_stats = perform_anova(small_df, 'value', 'group')
    assert 'error' in results
    assert group_stats.empty
