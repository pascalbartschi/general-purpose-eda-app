"""
Module for advanced exploratory data analysis functions of the EDA app.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
import umap

def calculate_correlation_with_pvalues(df: pd.DataFrame, 
                                     method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Calculate correlation matrix with p-values.
    
    Args:
        df: Input dataframe with numeric columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict]: (correlation_matrix, p_value_matrix, missing_data_info)
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Original counts of non-NaN values per column
    original_counts = df_numeric.count()
    
    # Calculate correlation matrix using pandas built-in method (handles NaN appropriately)
    corr_matrix = df_numeric.corr(method=method)
    
    # Initialize p-value matrix with NaN
    p_values = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
    
    # Dictionary to track dropped samples for each column pair
    dropped_info = {}
    
    # Calculate p-values
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col_i = df_numeric.columns[i]
            col_j = df_numeric.columns[j]
            
            # Find common indices where both columns have non-NaN values
            valid_indices = df_numeric[[col_i, col_j]].dropna().index
            
            # Skip if no valid indices
            if len(valid_indices) <= 1:
                p_values.iloc[i, j] = np.nan
                p_values.iloc[j, i] = np.nan
                continue
                
            # Extract values at valid indices
            values_i = df_numeric.loc[valid_indices, col_i]
            values_j = df_numeric.loc[valid_indices, col_j]
            
            # Store info about dropped samples
            total_samples = len(df_numeric)
            used_samples = len(valid_indices)
            dropped_samples = total_samples - used_samples
            
            if dropped_samples > 0:
                dropped_info[f"{col_i} vs {col_j}"] = {
                    "total_samples": total_samples,
                    "used_samples": used_samples,
                    "dropped_samples": dropped_samples,
                    "original_count_i": original_counts[col_i],
                    "original_count_j": original_counts[col_j],
                    "col_i": col_i,
                    "col_j": col_j
                }
            
            # Calculate correlation and p-value
            try:
                if method == 'pearson':
                    r, p = stats.pearsonr(values_i, values_j)
                elif method == 'spearman':
                    r, p = stats.spearmanr(values_i, values_j)
                elif method == 'kendall':
                    r, p = stats.kendalltau(values_i, values_j)
                else:
                    raise ValueError(f"Unsupported correlation method: {method}")
                
                p_values.iloc[i, j] = p
                p_values.iloc[j, i] = p  # p-value matrix is symmetric
            except Exception as e:
                p_values.iloc[i, j] = np.nan
                p_values.iloc[j, i] = np.nan
    
    # Set diagonal to 0
    np.fill_diagonal(p_values.values, 0)
    
    # Create summary of missing data
    missing_data_info = {
        "has_dropped_data": len(dropped_info) > 0,
        "dropped_pairs": dropped_info,
        "columns_with_most_missing": original_counts.sort_values().head(3).to_dict() if not original_counts.empty else {}
    }
    
    return corr_matrix, p_values, missing_data_info

def perform_pca(df: pd.DataFrame, 
               n_components: int = 2, 
               standardize: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Perform Principal Component Analysis (PCA).
    
    Args:
        df: Input dataframe with numeric columns
        n_components: Number of principal components to calculate
        standardize: Whether to standardize data before PCA
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]: (pca_results, loadings, explained_variance_ratio)
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Handle missing values by dropping rows with any NaN values
    df_numeric_clean = df_numeric.dropna()
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric_clean)
    else:
        scaled_data = df_numeric_clean.values
    
    # Perform PCA
    n_components = min(n_components, min(df_numeric.shape))
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create dataframe from results
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_results = pd.DataFrame(data=principal_components, columns=pc_columns, index=df_numeric_clean.index)
    
    # Calculate loadings (feature contributions to each component)
    loadings = pd.DataFrame(
        data=pca.components_.T,
        columns=pc_columns,
        index=df_numeric_clean.columns
    )
    
    return pca_results, loadings, pca.explained_variance_ratio_

def perform_tsne(df: pd.DataFrame, 
                perplexity: int = 30, 
                learning_rate: float = 200.0,
                n_iter: int = 1000) -> pd.DataFrame:
    """
    Perform t-SNE dimensionality reduction.
    
    Args:
        df: Input dataframe with numeric columns
        perplexity: t-SNE perplexity parameter
        learning_rate: t-SNE learning rate
        n_iter: Number of iterations for optimization
        
    Returns:
        pd.DataFrame: DataFrame with t-SNE results
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Handle missing values by dropping rows with any NaN values
    df_numeric_clean = df_numeric.dropna()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric_clean)
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, df_numeric.shape[0] - 1),  # Perplexity must be < n_samples - 1
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42
    )
    
    tsne_results = tsne.fit_transform(scaled_data)
    
    # Create dataframe from results with original indices preserved
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'], index=df_numeric_clean.index)
    
    return tsne_df

def perform_umap_reduction(df: pd.DataFrame, 
                         n_neighbors: int = 15, 
                         min_dist: float = 0.1) -> pd.DataFrame:
    """
    Perform UMAP dimensionality reduction.
    
    Args:
        df: Input dataframe with numeric columns
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        
    Returns:
        pd.DataFrame: DataFrame with UMAP results
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Handle missing values by dropping rows with any NaN values
    df_numeric_clean = df_numeric.dropna()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric_clean)
    
    # Perform UMAP
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, df_numeric.shape[0] - 1),
        min_dist=min_dist,
        random_state=42
    )
    
    umap_results = reducer.fit_transform(scaled_data)
    
    # Create dataframe from results with original indices preserved
    umap_df = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'], index=df_numeric_clean.index)
    
    return umap_df

def perform_kmeans(df: pd.DataFrame, 
                  n_clusters: int = 3) -> Tuple[pd.DataFrame, KMeans]:
    """
    Perform KMeans clustering.
    
    Args:
        df: Input dataframe with numeric columns
        n_clusters: Number of clusters
        
    Returns:
        Tuple[pd.DataFrame, KMeans]: (cluster_assignments, kmeans_model)
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Handle missing values by dropping rows with any NaN values
    df_numeric_clean = df_numeric.dropna()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric_clean)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create dataframe with cluster assignments and preserve original indices
    cluster_df = pd.DataFrame({
        'Cluster': clusters
    }, index=df_numeric_clean.index)
    
    return cluster_df, kmeans

def perform_dbscan(df: pd.DataFrame, 
                  eps: float = 0.5, 
                  min_samples: int = 5) -> pd.DataFrame:
    """
    Perform DBSCAN clustering.
    
    Args:
        df: Input dataframe with numeric columns
        eps: DBSCAN epsilon parameter (neighborhood distance threshold)
        min_samples: Minimum samples in neighborhood to form a core point
        
    Returns:
        pd.DataFrame: DataFrame with cluster assignments
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Handle missing values by dropping rows with any NaN values
    df_numeric_clean = df_numeric.dropna()
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric_clean)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    
    # Create dataframe with cluster assignments and preserve original indices
    cluster_df = pd.DataFrame({
        'Cluster': clusters
    }, index=df_numeric_clean.index)
    
    return cluster_df

def perform_ttest(df: pd.DataFrame, 
                 column: str, 
                 group_column: str) -> pd.DataFrame:
    """
    Perform t-test between groups.
    
    Args:
        df: Input dataframe
        column: Numeric column to test
        group_column: Categorical column defining groups
        
    Returns:
        pd.DataFrame: DataFrame with t-test results
    """
    # Get unique groups
    groups = df[group_column].unique()
    results = []
    
    # Perform pairwise t-tests
    for i in range(len(groups)):
        for j in range(i+1, len(groups)):
            group1 = groups[i]
            group2 = groups[j]
            
            values1 = df[df[group_column] == group1][column].dropna()
            values2 = df[df[group_column] == group2][column].dropna()
            
            # Skip if either group has too few values
            if len(values1) <= 1 or len(values2) <= 1:
                continue
            
            # Perform t-test
            t_stat, p_val = stats.ttest_ind(values1, values2, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            mean1, mean2 = values1.mean(), values2.mean()
            std1, std2 = values1.std(), values2.std()
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((len(values1) - 1) * std1**2 + (len(values2) - 1) * std2**2) / 
                              (len(values1) + len(values2) - 2))
            
            cohens_d = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else np.nan
            
            results.append({
                'Group 1': str(group1),
                'Group 2': str(group2),
                'Mean 1': mean1,
                'Mean 2': mean2,
                'Std Dev 1': std1,
                'Std Dev 2': std2,
                't-statistic': t_stat,
                'p-value': p_val,
                "Cohen's d": cohens_d,
                'Significant': p_val < 0.05
            })
    
    if not results:
        return pd.DataFrame()
        
    return pd.DataFrame(results)

def perform_anova(df: pd.DataFrame, 
                 column: str, 
                 group_column: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform one-way ANOVA test.
    
    Args:
        df: Input dataframe
        column: Numeric column to test
        group_column: Categorical column defining groups
        
    Returns:
        Tuple[Dict, pd.DataFrame]: (ANOVA results, group statistics)
    """
    # Get groups
    groups = []
    group_names = []
    
    for group in df[group_column].unique():
        values = df[df[group_column] == group][column].dropna()
        if len(values) > 0:
            groups.append(values)
            group_names.append(str(group))
    
    # Check if we have enough groups
    if len(groups) < 2:
        return {
            'error': 'Need at least 2 groups for ANOVA'
        }, pd.DataFrame()
    
    # Perform ANOVA
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Calculate eta-squared (effect size)
    # Sum of squares between groups
    grand_mean = np.mean([x for group in groups for x in group])
    ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
    
    # Sum of squares total
    all_values = np.concatenate(groups)
    ss_total = sum((x - grand_mean)**2 for x in all_values)
    
    eta_squared = ss_between / ss_total if ss_total != 0 else np.nan
    
    # Prepare group statistics
    group_stats = []
    for i, group in enumerate(groups):
        group_stats.append({
            'Group': group_names[i],
            'Count': len(group),
            'Mean': np.mean(group),
            'Std Dev': np.std(group),
            'Min': np.min(group),
            'Max': np.max(group)
        })
    
    result = {
        'f_statistic': f_stat,
        'p_value': p_val,
        'eta_squared': eta_squared,
        'significant': p_val < 0.05,
        'n_groups': len(groups),
        'n_samples': len(all_values)
    }
    
    return result, pd.DataFrame(group_stats)

def configure_plotly_figure(fig, height=3000, margin=None):
    """
    Configure a plotly figure with optimal display settings.
    
    Args:
        fig: Plotly figure object
        height: Figure height in pixels
        margin: Optional custom margin dictionary
        
    Returns:
        Updated plotly figure
    """
    # Set default margins if not provided
    if margin is None:
        margin = dict(l=80, r=80, t=100, b=80)
    
    # Update figure layout
    fig.update_layout(
        height=height,
        margin=margin,
        autosize=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(tickangle=-45),  # Angle tick labels to prevent overlap with long names
        font=dict(size=12)  # Slightly larger font
    )
    
    # For box plots specifically, check if we need to enhance them further
    if fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type == 'box':
        # Get axis labels
        if hasattr(fig.layout, 'xaxis') and hasattr(fig.layout.xaxis, 'title') and hasattr(fig.layout.xaxis.title, 'text'):
            x_label = fig.layout.xaxis.title.text
            # If we have categorical data on x-axis, check for long category names
            if x_label and len(x_label) > 15:
                fig.update_layout(
                    xaxis=dict(
                        tickangle=-90,  # Vertical labels for very long names
                        tickfont=dict(size=10)  # Slightly smaller font for long labels
                    ),
                    margin=dict(b=150)  # Extra bottom margin for vertical labels
                )
    
    return fig


def enhance_all_plotly_charts():
    """
    Monkeypatch Streamlit's plotly_chart function to automatically enhance all charts.
    This ensures all charts in the app have consistent and improved display properties.
    
    Applies improvements to all charts without having to modify each chart individually.
    """
    original_plotly_chart = st.plotly_chart
    
    def enhanced_plotly_chart(fig, *args, **kwargs):
        # Automatically enhance the figure if it's a plotly figure
        if hasattr(fig, 'update_layout'):
            # Apply size multipliers from session state if they exist
            height_multiplier = getattr(st.session_state, 'figure_height_multiplier', 1.0)
            width_multiplier = getattr(st.session_state, 'figure_width_multiplier', 1.0)
            
            # Default height based on complexity - use larger default height
            height = 800  # Increased default height
            if hasattr(fig, 'data') and len(fig.data) > 0:
                # More data traces = taller figure
                height = max(800, 600 + 60 * len(fig.data))  # Increased base and multiplier
            
            # Apply user's height multiplier
            height = int(height * height_multiplier)
            
            # For heatmaps (correlation plots, etc.)
            if fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type in ['heatmap', 'imshow']:
                if hasattr(fig.data[0], 'z') and hasattr(fig.data[0].z, 'shape'):
                    # Size based on matrix dimensions
                    rows, cols = fig.data[0].z.shape
                    # Make correlation matrices much larger
                    height = max(900, 400 + 60 * rows)  # Increased to ensure readability
                    # Apply user's height multiplier
                    height = int(height * height_multiplier)
                    # Adjust margins to give more space to the actual heatmap
                    margin = dict(l=150, r=150, t=100, b=150)  # Increased margins
                    fig = configure_plotly_figure(fig, height=height, margin=margin)
                    
                    # Set width explicitly for heatmaps to prevent labels from taking over
                    width = 1000 * width_multiplier
                    fig.update_layout(width=width)
            
            # For scatter plots
            elif fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type == 'scatter':
                # Default configuration with user's height multiplier
                fig = configure_plotly_figure(fig, height=int(600 * height_multiplier))
                # Set width for scatter plots
                width = 900 * width_multiplier
                fig.update_layout(width=width)
            
            # For box plots
            elif fig.data and hasattr(fig.data[0], 'type') and fig.data[0].type == 'box':
                # Special handling for box plots with more bottom margin
                fig = configure_plotly_figure(fig, height=int(600 * height_multiplier), 
                                           margin=dict(l=80, r=80, t=100, b=120))
                # Set width for box plots
                width = 900 * width_multiplier
                fig.update_layout(width=width)
            
            # For all other plot types, apply standard configuration
            else:
                fig = configure_plotly_figure(fig, height=int(600 * height_multiplier))
                # Set width for other plots
                width = 800 * width_multiplier
                fig.update_layout(width=width)
            
        # Call the original function with our enhanced figure
        return original_plotly_chart(fig, *args, **kwargs)
    
    # Replace Streamlit's plotly_chart with our enhanced version
    st.plotly_chart = enhanced_plotly_chart

def check_missing_data(df: pd.DataFrame, selected_cols: List[str]) -> Dict:
    """
    Check for missing data in selected columns and prepare a report.
    
    Args:
        df: Input dataframe
        selected_cols: List of column names to check
        
    Returns:
        Dict: Missing data information with counts and percentages
    """
    # Extract just the columns we're interested in
    df_subset = df[selected_cols]
    
    # Count missing values per column
    missing_counts = df_subset.isna().sum()
    
    # Count rows with any missing values
    rows_with_missing = df_subset.isna().any(axis=1).sum()
    
    # Complete rows (no missing values)
    complete_rows = len(df_subset) - rows_with_missing
    
    # Calculate percentage of missing data
    missing_percentages = (missing_counts / len(df_subset) * 100).round(2)
    
    # Find columns with highest missing values
    columns_most_missing = missing_counts.sort_values(ascending=False).head(3).to_dict()
    
    # Prepare the report
    report = {
        "has_missing_data": rows_with_missing > 0,
        "total_rows": len(df_subset),
        "rows_with_missing": rows_with_missing,
        "complete_rows": complete_rows,
        "missing_percentage": (rows_with_missing / len(df_subset) * 100).round(2),
        "column_missing_counts": missing_counts.to_dict(),
        "column_missing_percentages": missing_percentages.to_dict(),
        "columns_most_missing": columns_most_missing
    }
    
    return report


def display_missing_data_warning(report: Dict) -> None:
    """
    Display a warning and detailed information about missing data.
    
    Args:
        report: Missing data report from check_missing_data function
    """
    if report["has_missing_data"]:
        st.warning(
            f"{report['rows_with_missing']} rows ({report['missing_percentage']:.1f}% of data) "
            f"contain missing values and will be excluded from analysis. "
            f"Only {report['complete_rows']} complete rows will be used."
        )
        
        # Show detailed info in an expander
        with st.expander("View details on missing data"):
            st.markdown("**Missing data summary:**")
            st.markdown(
                f"- Total rows: {report['total_rows']}\n"
                f"- Rows with missing values: {report['rows_with_missing']} ({report['missing_percentage']:.1f}%)\n"
                f"- Complete rows: {report['complete_rows']} ({100-report['missing_percentage']:.1f}%)"
            )
            
            st.markdown("**Missing values by column:**")
            for col, count in report["column_missing_counts"].items():
                if count > 0:
                    percentage = report["column_missing_percentages"][col]
                    st.markdown(f"- {col}: {count} missing values ({percentage:.1f}%)")
                    
            # Provide some advice
            if report["missing_percentage"] > 30:
                st.markdown("⚠️ **High proportion of missing data may affect results significantly.**")
                st.markdown("Consider using imputation techniques or removing columns with too many missing values.")

def advanced_eda_ui(df: pd.DataFrame) -> None:
    """
    Streamlit UI for advanced exploratory data analysis.
    
    Args:
        df: Input dataframe
    """
    # Enable enhanced charts for better visualization
    enhance_all_plotly_charts()
    
    st.title("Advanced Exploratory Data Analysis")
    
    # Add figure size controls in an expander
    with st.sidebar.expander("Figure Size Settings", expanded=False):
        st.caption("Adjust visualization sizes to improve readability")
        
        # Store settings in session state to persist between reruns
        if "figure_height_multiplier" not in st.session_state:
            st.session_state.figure_height_multiplier = 1.0
        if "figure_width_multiplier" not in st.session_state:
            st.session_state.figure_width_multiplier = 1.0
            
        # Let user adjust figure size multipliers
        height_mult = st.slider("Height multiplier", 0.5, 2.0, 
                               st.session_state.figure_height_multiplier, 0.1,
                               help="Increase to make figures taller")
        width_mult = st.slider("Width multiplier", 0.5, 2.0, 
                              st.session_state.figure_width_multiplier, 0.1,
                              help="Increase to make figures wider")
        
        # Save settings to session state
        if height_mult != st.session_state.figure_height_multiplier:
            st.session_state.figure_height_multiplier = height_mult
        if width_mult != st.session_state.figure_width_multiplier:
            st.session_state.figure_width_multiplier = width_mult
    
    if df is None or df.empty:
        st.warning("Please upload and clean a dataset first.")
        return
    
    # Create tabs for different advanced analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "Correlation Analysis", 
        "Dimensionality Reduction", 
        "Clustering", 
        "Hypothesis Testing"
    ])
    
    # Tab 1: Correlation Analysis with p-values
    with tab1:
        st.subheader("Correlation Analysis with Statistical Significance")
        
        # Get numeric columns
        # Get numeric columns that are not always None/NaN
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                        if not df[col].isna().all()]
        




        
        if len(numeric_cols) > 1:
            # Select columns for correlation
            selected_cols = st.multiselect(
                "Select columns for correlation analysis:",
                options=numeric_cols,
                default=numeric_cols[:min(10, len(numeric_cols))]
            )
            
            if selected_cols and len(selected_cols) > 1:
                # Select correlation method
                corr_method = st.radio(
                    "Select correlation method:",
                    ["Pearson", "Spearman", "Kendall"],
                    horizontal=True
                )
                
                # Calculate correlation and p-values
                corr_matrix, p_values, missing_data_info = calculate_correlation_with_pvalues(
                    df[selected_cols], 
                    method=corr_method.lower()
                )
                
                # Display warning if data was dropped due to missing values
                if missing_data_info["has_dropped_data"]:
                    st.warning(
                        "Some rows were excluded from correlation analysis due to missing values. "
                        "Each correlation pair uses only rows where both variables have non-missing values."
                    )
                    
                    # Show detailed info in an expander
                    with st.expander("View details on missing data"):
                        st.markdown("**Column pairs with missing data:**")
                        
                        for pair, info in missing_data_info["dropped_pairs"].items():
                            col_i, col_j = info["col_i"], info["col_j"]
                            st.markdown(
                                f"- **{pair}**: Used {info['used_samples']} out of {info['total_samples']} rows "
                                f"({info['dropped_samples']} rows excluded). "
                                f"Original non-missing counts: {col_i}: {info['original_count_i']}, "
                                f"{col_j}: {info['original_count_j']}"
                            )
                        
                        if missing_data_info["columns_with_most_missing"]:
                            st.markdown("**Columns with most missing values:**")
                            for col, count in missing_data_info["columns_with_most_missing"].items():
                                st.markdown(f"- {col}: {len(df) - count} missing values")
                
                # Display correlation heatmap
                st.write(f"**{corr_method} Correlation Heatmap:**")
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation"
                )
                # Ensure figure is displayed with plenty of space for long column names
                # Significantly increase height for better visualization
                fig = configure_plotly_figure(fig, 
                                           height=max(1000, 500 + 70 * len(corr_matrix.columns)),
                                           margin=dict(l=150, r=150, t=100, b=150))
                
                # Make the actual visualization area larger relative to the labels
                fig.update_layout(
                    width=1200,  # Set a fixed width
                    height=max(1000, 500 + 70 * len(corr_matrix.columns))  # Match height from above
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display p-values heatmap
                st.write("**P-values Heatmap:**")
                # Set NaN diagonal values to 1 for visualization
                p_values_vis = p_values.copy()
                np.fill_diagonal(p_values_vis.values, 1)
                
                fig_p = px.imshow(
                    p_values_vis,
                    text_auto=".3f",
                    color_continuous_scale="Viridis_r",  # Reversed so dark = small p-value
                    title=f"P-values for {corr_method} Correlation"
                )
                # Apply consistent sizing to p-values heatmap
                # Significantly increase height for better visualization
                fig_p = configure_plotly_figure(fig_p, 
                                             height=max(1000, 500 + 70 * len(p_values_vis.columns)),
                                             margin=dict(l=150, r=150, t=100, b=150))
                
                # Make the actual visualization area larger relative to the labels
                fig_p.update_layout(
                    width=1200,  # Set a fixed width
                    height=max(1000, 500 + 70 * len(p_values_vis.columns))  # Match height from above
                )
                
                st.plotly_chart(fig_p, use_container_width=True)
                
                # Display significant correlations
                st.write("**Significant Correlations (p < 0.05):**")
                
                # Create table of significant correlations
                significant_corrs = []
                
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        # Check if p-value and correlation are valid (not NaN) and significant
                        if not pd.isna(p_values.iloc[i, j]) and p_values.iloc[i, j] < 0.05 and not pd.isna(corr_matrix.iloc[i, j]):
                            significant_corrs.append({
                                'Variable 1': selected_cols[i],
                                'Variable 2': selected_cols[j],
                                f'{corr_method} Correlation': corr_matrix.iloc[i, j],
                                'p-value': p_values.iloc[i, j],
                                'Strength': 'Strong' if abs(corr_matrix.iloc[i, j]) > 0.7 else 
                                           'Moderate' if abs(corr_matrix.iloc[i, j]) > 0.3 else 'Weak'
                            })
                
                if significant_corrs:
                    significant_df = pd.DataFrame(significant_corrs)
                    st.dataframe(significant_df.sort_values(f'{corr_method} Correlation', key=abs, ascending=False))
                else:
                    st.info("No statistically significant correlations found.")
                
    
    # Tab 2: Dimensionality Reduction
    with tab2:
        st.subheader("Dimensionality Reduction")
        
        # Get numeric columns
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
                if not df[col].isna().all()]

        
        if len(numeric_cols) >= 2:
            # Select columns for dimensionality reduction
            selected_cols = st.multiselect(
                "Select columns for dimensionality reduction:",
                options=numeric_cols,
                default=numeric_cols
            )
            
            if len(selected_cols) >= 2:
                # Select method
                reduction_method = st.selectbox(
                    "Select dimensionality reduction method:",
                    ["PCA", "t-SNE", "UMAP"]
                )
                
                # Select column for coloring points
                all_cols = df.columns.tolist()
                all_cols.insert(0, None)  # Add None as first option
                color_by = st.selectbox("Color points by:", all_cols)
                
                if reduction_method == "PCA":
                    # PCA specific parameters
                    n_components = st.slider("Number of components:", 2, min(10, len(selected_cols)), 2)
                    standardize = st.checkbox("Standardize data", value=True)
                    
                    # Check for missing data
                    missing_data_report = check_missing_data(df, selected_cols)
                    display_missing_data_warning(missing_data_report)
                    
                    # Perform PCA
                    try:
                        pca_results, loadings, explained_var = perform_pca(
                            df[selected_cols], 
                            n_components=n_components,
                            standardize=standardize
                        )
                        
                        # Plot first two components
                        # For coloring, we need to use the same indices as the PCA results
                        # Only use the color by column for the rows that actually made it into the PCA results
                        color_data = df.loc[pca_results.index, color_by] if color_by else None
                        
                        fig = px.scatter(
                            pca_results, 
                            x='PC1', 
                            y='PC2',
                            title=f"PCA: First two components explain {100 * sum(explained_var[:2]):.1f}% of variance",
                            color=color_data
                        )
                        # Configure PCA scatter plot
                        fig = configure_plotly_figure(fig, height=600)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show explained variance
                        st.write("**Explained Variance Ratio:**")
                        
                        explained_df = pd.DataFrame({
                            'Component': [f"PC{i+1}" for i in range(len(explained_var))],
                            'Explained Variance (%)': [100 * var for var in explained_var],
                            'Cumulative Variance (%)': [100 * sum(explained_var[:i+1]) for i in range(len(explained_var))]
                        })
                        
                        st.dataframe(explained_df)
                        
                        # Show feature loadings
                        st.write("**Feature Contributions (Loadings):**")
                        st.dataframe(loadings)
                        
                        # Visualization of loadings for first two components
                        fig_loadings = px.scatter(
                            x=loadings['PC1'],
                            y=loadings['PC2'],
                            text=loadings.index,
                            title="Feature Loadings"
                        )
                        
                        fig_loadings.update_traces(textposition='top center')
                        fig_loadings.update_layout(
                            xaxis_title="PC1 Loading",
                            yaxis_title="PC2 Loading",
                            xaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black'),
                            yaxis=dict(zeroline=True, zerolinewidth=1, zerolinecolor='black')
                        )
                        
                        # Configure loadings plot with additional space for text labels
                        fig_loadings = configure_plotly_figure(fig_loadings, 
                                                           height=600,
                                                           margin=dict(l=80, r=80, t=100, b=80))
                        # Adjust to ensure text labels are visible
                        if len(loadings) > 10:
                            fig_loadings.update_layout(height=700)
                            
                        st.plotly_chart(fig_loadings, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error performing PCA: {str(e)}")
                
                elif reduction_method == "t-SNE":
                    # t-SNE specific parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        perplexity = st.slider(
                            "Perplexity:", 
                            5, 
                            min(50, len(df) - 1), 
                            30
                        )
                    
                    with col2:
                        learning_rate = st.slider("Learning rate:", 10.0, 500.0, 200.0, 10.0)
                    
                    n_iter = st.slider("Number of iterations:", 250, 2000, 1000, 250)
                    
                    # Check for missing data
                    missing_data_report = check_missing_data(df, selected_cols)
                    display_missing_data_warning(missing_data_report)
                    
                    # Perform t-SNE
                    if st.button("Run t-SNE"):
                        try:
                            with st.spinner("Running t-SNE (this may take a while)..."):
                                tsne_results = perform_tsne(
                                    df[selected_cols],
                                    perplexity=perplexity,
                                    learning_rate=learning_rate,
                                    n_iter=n_iter
                                )
                                
                                # Plot t-SNE results
                                # For coloring, we need to use the same indices as the t-SNE results
                                # Only use the color by column for the rows that actually made it into the t-SNE results
                                color_data = df.loc[tsne_results.index, color_by] if color_by else None
                                
                                fig = px.scatter(
                                    tsne_results, 
                                    x='TSNE1', 
                                    y='TSNE2',
                                    title="t-SNE Projection",
                                    color=color_data
                                )
                                # Configure t-SNE plot
                                fig = configure_plotly_figure(fig, height=650)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.success("t-SNE completed successfully!")
                        except Exception as e:
                            st.error(f"Error performing t-SNE: {str(e)}")
                
                elif reduction_method == "UMAP":
                    # UMAP specific parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        n_neighbors = st.slider(
                            "Number of neighbors:", 
                            2, 
                            min(100, len(df) - 1), 
                            15
                        )
                    
                    with col2:
                        min_dist = st.slider("Minimum distance:", 0.0, 1.0, 0.1, 0.05)
                    
                    # Check for missing data
                    missing_data_report = check_missing_data(df, selected_cols)
                    display_missing_data_warning(missing_data_report)
                    
                    # Perform UMAP
                    if st.button("Run UMAP"):
                        try:
                            with st.spinner("Running UMAP (this may take a while)..."):
                                umap_results = perform_umap_reduction(
                                    df[selected_cols],
                                    n_neighbors=n_neighbors,
                                    min_dist=min_dist
                                )
                                
                                # Plot UMAP results
                                # For coloring, we need to use the same indices as the UMAP results
                                # Only use the color by column for the rows that actually made it into the UMAP results
                                color_data = df.loc[umap_results.index, color_by] if color_by else None
                                
                                fig = px.scatter(
                                    umap_results, 
                                    x='UMAP1', 
                                    y='UMAP2',
                                    title="UMAP Projection",
                                    color=color_data
                                )
                                # Configure UMAP plot
                                fig = configure_plotly_figure(fig, height=650)
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.success("UMAP completed successfully!")
                        except Exception as e:
                            st.error(f"Error performing UMAP: {str(e)}")
            else:
                st.warning("Please select at least 2 columns for dimensionality reduction.")
        else:
            st.warning("Need at least 2 numeric columns for dimensionality reduction.")
    
    # Tab 3: Clustering
    with tab3:
        st.subheader("Clustering Analysis")
        
        # Get numeric columns
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
        if not df[col].isna().all()]
        
        if len(numeric_cols) >= 2:
            # Select columns for clustering
            selected_cols = st.multiselect(
                "Select columns for clustering:",
                options=numeric_cols,
                default=numeric_cols
            )
            
            if len(selected_cols) >= 2:
                # Select clustering method
                clustering_method = st.selectbox(
                    "Select clustering method:",
                    ["K-Means", "DBSCAN"]
                )
                
                if clustering_method == "K-Means":
                    # K-Means specific parameters
                    n_clusters = st.slider("Number of clusters:", 2, 10, 3)
                    
                    # Check for missing data
                    missing_data_report = check_missing_data(df, selected_cols)
                    display_missing_data_warning(missing_data_report)
                    
                    # Perform K-Means
                    if st.button("Run K-Means"):
                        try:
                            cluster_assignments, kmeans_model = perform_kmeans(
                                df[selected_cols],
                                n_clusters=n_clusters
                            )
                            
                            # Get the data with only the rows actually used in clustering (no NaNs)
                            # Take only rows that were actually clustered (those with indices in cluster_assignments)
                            filtered_df = df.loc[cluster_assignments.index]
                            
                            # Add cluster assignments to dataframe for visualization
                            vis_df = filtered_df[selected_cols].copy()
                            vis_df['Cluster'] = cluster_assignments['Cluster']
                            
                            # Perform PCA for visualization if needed
                            if len(selected_cols) > 2:
                                # Use the filtered_df that contains only rows without NaNs
                                pca_results, _, _ = perform_pca(filtered_df[selected_cols])
                                vis_df['PC1'] = pca_results['PC1']
                                vis_df['PC2'] = pca_results['PC2']
                                x_col, y_col = 'PC1', 'PC2'
                                title = "K-Means Clusters (PCA projection)"
                            else:
                                x_col, y_col = selected_cols[0], selected_cols[1]
                                title = f"K-Means Clusters: {y_col} vs {x_col}"
                            
                            # Plot clusters
                            fig = px.scatter(
                                vis_df,
                                x=x_col,
                                y=y_col,
                                color='Cluster',
                                title=title,
                                color_continuous_scale=px.colors.qualitative.G10
                            )
                            # Configure clustering plot
                            fig = configure_plotly_figure(fig, height=650)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster centers if PCA was used
                            if len(selected_cols) > 2:
                                # Filter out rows with NaN values - use the same data that was used for clustering
                                df_for_pca = df[selected_cols].dropna()
                                
                                # Transform cluster centers to PCA space
                                centers_pca = StandardScaler().fit_transform(kmeans_model.cluster_centers_)
                                pca = PCA(n_components=2)
                                # Fit PCA on the same clean data used for clustering
                                pca.fit(df_for_pca)
                                centers_pca = pca.transform(centers_pca)
                                
                                # Add cluster centers to plot
                                fig.add_trace(
                                    go.Scatter(
                                        x=centers_pca[:, 0],
                                        y=centers_pca[:, 1],
                                        mode='markers',
                                        marker=dict(
                                            symbol='x',
                                            color='black',
                                            size=16,  # Larger cluster center markers
                                            line=dict(width=2)
                                        ),
                                        name='Cluster Centers'
                                    )
                                )
                                # Configure with cluster centers
                                fig = configure_plotly_figure(fig, height=650)
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster statistics
                            st.write("**Cluster Statistics:**")
                            
                            cluster_stats = []
                            for i in range(n_clusters):
                                cluster_data = vis_df[vis_df['Cluster'] == i]
                                cluster_stats.append({
                                    'Cluster': i,
                                    'Size': len(cluster_data),
                                    'Size (%)': f"{100 * len(cluster_data) / len(vis_df):.1f}%"
                                })
                                
                                # Add feature means for each cluster
                                for col in selected_cols:
                                    if pd.api.types.is_numeric_dtype(vis_df[col]):
                                        cluster_stats[-1][f'Mean {col}'] = cluster_data[col].mean()
                                        cluster_stats[-1][f'Std {col}'] = cluster_data[col].std()
                            
                            st.dataframe(pd.DataFrame(cluster_stats))
                        except Exception as e:
                            st.error(f"Error performing K-Means: {str(e)}")
                
                elif clustering_method == "DBSCAN":
                    # DBSCAN specific parameters
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        eps = st.slider("Epsilon (neighborhood distance):", 0.1, 2.0, 0.5, 0.1)
                    
                    with col2:
                        min_samples = st.slider("Minimum samples in neighborhood:", 2, 20, 5)
                    
                    # Check for missing data
                    missing_data_report = check_missing_data(df, selected_cols)
                    display_missing_data_warning(missing_data_report)
                    
                    # Perform DBSCAN
                    if st.button("Run DBSCAN"):
                        try:
                            cluster_assignments = perform_dbscan(
                                df[selected_cols],
                                eps=eps,
                                min_samples=min_samples
                            )
                            
                            # Get the data with only the rows actually used in clustering (no NaNs)
                            # Take only rows that were actually clustered (those with indices in cluster_assignments)
                            filtered_df = df.loc[cluster_assignments.index]
                            
                            # Add cluster assignments to dataframe for visualization
                            vis_df = filtered_df[selected_cols].copy()
                            vis_df['Cluster'] = cluster_assignments['Cluster']
                            
                            # Perform PCA for visualization if needed
                            if len(selected_cols) > 2:
                                # Use the filtered_df that contains only rows without NaNs
                                pca_results, _, _ = perform_pca(filtered_df[selected_cols])
                                vis_df['PC1'] = pca_results['PC1']
                                vis_df['PC2'] = pca_results['PC2']
                                x_col, y_col = 'PC1', 'PC2'
                                title = "DBSCAN Clusters (PCA projection)"
                            else:
                                x_col, y_col = selected_cols[0], selected_cols[1]
                                title = f"DBSCAN Clusters: {y_col} vs {x_col}"
                            
                            # Plot clusters
                            fig = px.scatter(
                                vis_df,
                                x=x_col,
                                y=y_col,
                                color='Cluster',
                                title=title,
                                color_continuous_scale=px.colors.qualitative.G10
                            )
                            
                            # Configure DBSCAN plot
                            fig = configure_plotly_figure(fig, height=650)
                            
                            # If there are outliers, make them more visible
                            if -1 in vis_df['Cluster'].unique():
                                # Find the color index for -1
                                outlier_color = 'black'
                                fig.update_traces(
                                    marker=dict(
                                        size=8,  # Slightly larger markers for better visibility
                                        opacity=0.7
                                    )
                                )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster statistics
                            st.write("**Cluster Statistics:**")
                            
                            # Count cluster sizes
                            cluster_counts = vis_df['Cluster'].value_counts().reset_index()
                            cluster_counts.columns = ['Cluster', 'Count']
                            
                            # Add percentage column
                            cluster_counts['Percentage'] = 100 * cluster_counts['Count'] / len(vis_df)
                            
                            st.dataframe(cluster_counts)
                            
                            # Show outliers (cluster -1)
                            if -1 in vis_df['Cluster'].values:
                                st.write(f"**Outliers detected:** {len(vis_df[vis_df['Cluster'] == -1])} points")
                        except Exception as e:
                            st.error(f"Error performing DBSCAN: {str(e)}")
            else:
                st.warning("Please select at least 2 columns for clustering.")
        else:
            st.warning("Need at least 2 numeric columns for clustering.")
    
    # Tab 4: Hypothesis Testing
    with tab4:
        st.subheader("Hypothesis Testing")
        
        # Get numeric and categorical columns
        numeric_cols = [col for col in df.select_dtypes(include=['number']).columns 
        if not df[col].isna().all()]
        categorical_cols = [col for col in df.select_dtypes(exclude=['number']).columns 
        if not df[col].isna().all()]
        
        if numeric_cols and categorical_cols:
            # Select columns for test
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_col = st.selectbox(
                    "Select numeric variable:",
                    options=numeric_cols
                )
            
            with col2:
                categorical_col = st.selectbox(
                    "Select categorical variable (groups):",
                    options=categorical_cols
                )
                
            # Auto-determine the appropriate test based on number of unique values in categorical variable
            if categorical_col:
                n_unique_groups = len(df[categorical_col].dropna().unique())
                
                if n_unique_groups < 2:
                    st.warning("The selected categorical variable needs at least 2 groups for hypothesis testing.")
                    test_type = None
                elif n_unique_groups == 2:
                    test_type = "T-Test"
                    st.info(f"✅ **T-Test automatically selected** because '{categorical_col}' has exactly 2 groups.")
                else:
                    test_type = "ANOVA"
                    st.info(f"✅ **ANOVA automatically selected** because '{categorical_col}' has {n_unique_groups} groups.")
                
                # Display the unique values/groups that will be compared
                unique_values = df[categorical_col].dropna().unique()
                st.write(f"**Groups to compare:** {', '.join(str(val) for val in unique_values)}")
            else:
                test_type = None
            
            if test_type == "T-Test":
                # Perform T-Test
                if st.button("Run T-Test Analysis"):
                    try:
                        results = perform_ttest(df, numeric_col, categorical_col)
                        
                        if not results.empty:
                            # Show test results
                            st.write("**T-Test Results:**")
                            st.dataframe(results)
                            
                            # Visualize the groups
                            st.write("**Group Comparison:**")
                            
                            fig = px.box(
                                df,
                                x=categorical_col,
                                y=numeric_col,
                                title=f"{numeric_col} by {categorical_col}",
                                points="all"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Insufficient data for T-Test. Ensure you have at least two groups with sufficient data.")
                    except Exception as e:
                        st.error(f"Error performing T-Test: {str(e)}")
            
            elif test_type == "ANOVA":
                # Perform ANOVA
                if st.button("Run ANOVA Analysis"):
                    try:
                        anova_results, group_stats = perform_anova(df, numeric_col, categorical_col)
                        
                        if 'error' not in anova_results:
                            # Show ANOVA results
                            st.write("**ANOVA Results:**")
                            
                            result_df = pd.DataFrame({
                                'F-statistic': [anova_results['f_statistic']],
                                'p-value': [anova_results['p_value']],
                                'Eta-squared': [anova_results['eta_squared']],
                                'Significant': [anova_results['significant']],
                                'Groups': [anova_results['n_groups']],
                                'Total Samples': [anova_results['n_samples']]
                            })
                            
                            st.dataframe(result_df)
                            
                            # Show interpretation
                            if anova_results['significant']:
                                st.success(f"The ANOVA test is statistically significant (p < 0.05). There are significant differences in {numeric_col} between groups of {categorical_col}.")
                                
                                # Effect size interpretation
                                eta_sq = anova_results['eta_squared']
                                if eta_sq < 0.06:
                                    effect = "small"
                                elif eta_sq < 0.14:
                                    effect = "medium"
                                else:
                                    effect = "large"
                                
                                st.write(f"The effect size (Eta-squared = {eta_sq:.3f}) indicates a {effect} effect.")
                            else:
                                st.info(f"The ANOVA test is not statistically significant (p > 0.05). There are no significant differences in {numeric_col} between groups of {categorical_col}.")
                            
                            # Show group statistics
                            st.write("**Group Statistics:**")
                            st.dataframe(group_stats)
                            
                            # Visualize the groups
                            st.write("**Group Comparison:**")
                            
                            fig = px.box(
                                df,
                                x=categorical_col,
                                y=numeric_col,
                                title=f"{numeric_col} by {categorical_col}",
                                points="all"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning(anova_results['error'])
                    except Exception as e:
                        st.error(f"Error performing ANOVA: {str(e)}")
                        
            elif test_type is None:
                st.warning("Please select a categorical column with at least 2 groups to perform hypothesis testing.")
        else:
            st.warning("Need both numeric and categorical columns for hypothesis testing.")
        
        # Add information about hypothesis tests
            with st.expander("ℹ️ About Statistical Tests", expanded=False):
                st.markdown("""
                **Statistical Test Selection:**
                - **T-Test** is used to compare means between exactly **two groups** (binary categorical variables)
                - **ANOVA** (Analysis of Variance) is used when comparing means across **three or more groups**
                
                The appropriate test is automatically selected based on the number of unique values in your chosen categorical variable.
                
                **Understanding p-values:**
                - p < 0.05 indicates statistically significant differences between groups
                - Effect size measures the strength of the relationship (small, medium, or large)
                """)
