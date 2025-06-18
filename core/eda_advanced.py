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
                                     method: str = 'pearson') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix with p-values.
    
    Args:
        df: Input dataframe with numeric columns
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (correlation_matrix, p_value_matrix)
    """
    # Ensure we only use numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = df_numeric.corr(method=method)
    
    # Initialize p-value matrix with NaN
    p_values = pd.DataFrame(np.nan, index=corr_matrix.index, columns=corr_matrix.columns)
    
    # Calculate p-values
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if method == 'pearson':
                r, p = stats.pearsonr(df_numeric.iloc[:, i].dropna(), df_numeric.iloc[:, j].dropna())
            elif method == 'spearman':
                r, p = stats.spearmanr(df_numeric.iloc[:, i].dropna(), df_numeric.iloc[:, j].dropna())
            elif method == 'kendall':
                r, p = stats.kendalltau(df_numeric.iloc[:, i].dropna(), df_numeric.iloc[:, j].dropna())
            else:
                raise ValueError(f"Unsupported correlation method: {method}")
                
            p_values.iloc[i, j] = p
            p_values.iloc[j, i] = p  # p-value matrix is symmetric
    
    # Set diagonal to 0
    np.fill_diagonal(p_values.values, 0)
    
    return corr_matrix, p_values

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
    
    # Standardize if requested
    if standardize:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_numeric)
    else:
        scaled_data = df_numeric.values
    
    # Perform PCA
    n_components = min(n_components, min(df_numeric.shape))
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create dataframe from results
    pc_columns = [f"PC{i+1}" for i in range(n_components)]
    pca_results = pd.DataFrame(data=principal_components, columns=pc_columns)
    
    # Calculate loadings (feature contributions to each component)
    loadings = pd.DataFrame(
        data=pca.components_.T,
        columns=pc_columns,
        index=df_numeric.columns
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
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=min(perplexity, df_numeric.shape[0] - 1),  # Perplexity must be < n_samples - 1
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=42
    )
    
    tsne_results = tsne.fit_transform(scaled_data)
    
    # Create dataframe from results
    tsne_df = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
    
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
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Perform UMAP
    reducer = umap.UMAP(
        n_neighbors=min(n_neighbors, df_numeric.shape[0] - 1),
        min_dist=min_dist,
        random_state=42
    )
    
    umap_results = reducer.fit_transform(scaled_data)
    
    # Create dataframe from results
    umap_df = pd.DataFrame(data=umap_results, columns=['UMAP1', 'UMAP2'])
    
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
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Create dataframe with cluster assignments
    cluster_df = pd.DataFrame({
        'Cluster': clusters
    })
    
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
    
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    
    # Perform DBSCAN clustering
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(scaled_data)
    
    # Create dataframe with cluster assignments
    cluster_df = pd.DataFrame({
        'Cluster': clusters
    })
    
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

def advanced_eda_ui(df: pd.DataFrame) -> None:
    """
    Streamlit UI for advanced exploratory data analysis.
    
    Args:
        df: Input dataframe
    """
    st.title("Advanced Exploratory Data Analysis")
    
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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
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
                corr_matrix, p_values = calculate_correlation_with_pvalues(
                    df[selected_cols], 
                    method=corr_method.lower()
                )
                
                # Display correlation heatmap
                st.write(f"**{corr_method} Correlation Heatmap:**")
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title=f"{corr_method} Correlation"
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
                st.plotly_chart(fig_p, use_container_width=True)
                
                # Display significant correlations
                st.write("**Significant Correlations (p < 0.05):**")
                
                # Create table of significant correlations
                significant_corrs = []
                
                for i in range(len(selected_cols)):
                    for j in range(i+1, len(selected_cols)):
                        if p_values.iloc[i, j] < 0.05:
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
                
                # Correlation vs. Causation Helper
                st.subheader("Correlation vs. Causation Helper")
                st.write("""
                Remember that correlation does not imply causation. Consider these potential confounding explanations:
                
                1. **Common Cause**: Both variables might be affected by a third variable
                2. **Reverse Causality**: The direction of cause and effect might be reversed
                3. **Coincidence**: Especially with small samples, correlations can occur by chance
                4. **Indirect Relationship**: Variables may be connected through a chain of relationships
                
                Consider stratifying your data by potential confounding variables to check if correlations hold across different groups.
                """)
                
                # Allow selection of variables and potential confounders
                if len(significant_corrs) > 0 and len(df.columns) > 2:
                    st.write("**Explore Potential Confounding:**")
                    
                    var_options = [f"{row['Variable 1']} vs {row['Variable 2']}" for row in significant_corrs]
                    selected_pair = st.selectbox("Select a correlation to explore:", var_options)
                    
                    if selected_pair:
                        var1, var2 = selected_pair.split(" vs ")
                        
                        # Select potential confounder
                        other_cols = [col for col in df.columns if col not in [var1, var2]]
                        confounder = st.selectbox("Select potential confounder:", other_cols)
                        
                        if pd.api.types.is_numeric_dtype(df[confounder]):
                            # For numeric confounders, create segments
                            n_bins = st.slider("Number of bins for confounder:", 2, 5, 3)
                            
                            df['confounder_bin'] = pd.qcut(df[confounder], q=n_bins, duplicates='drop')
                            
                            # Create scatter plot colored by bins
                            fig = px.scatter(
                                df, 
                                x=var1,
                                y=var2,
                                color='confounder_bin',
                                title=f"{var2} vs {var1}, stratified by {confounder}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show correlations within each bin
                            st.write("**Correlations within strata:**")
                            
                            strata_corrs = []
                            for bin_val in df['confounder_bin'].unique():
                                bin_df = df[df['confounder_bin'] == bin_val]
                                if len(bin_df) > 5:  # Need sufficient data for correlation
                                    bin_corr = bin_df[[var1, var2]].corr(method=corr_method.lower()).iloc[0, 1]
                                    strata_corrs.append({
                                        'Stratum': str(bin_val),
                                        'Correlation': bin_corr,
                                        'Sample Size': len(bin_df)
                                    })
                            
                            if strata_corrs:
                                st.dataframe(pd.DataFrame(strata_corrs))
                        else:
                            # For categorical confounders
                            confounder_vals = df[confounder].unique()
                            
                            if len(confounder_vals) <= 10:  # Only if reasonably few categories
                                # Create scatter plot colored by category
                                fig = px.scatter(
                                    df, 
                                    x=var1,
                                    y=var2,
                                    color=confounder,
                                    title=f"{var2} vs {var1}, colored by {confounder}"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Show correlations within each category
                                st.write("**Correlations within groups:**")
                                
                                group_corrs = []
                                for group in confounder_vals:
                                    group_df = df[df[confounder] == group]
                                    if len(group_df) > 5:  # Need sufficient data for correlation
                                        group_corr = group_df[[var1, var2]].corr(method=corr_method.lower()).iloc[0, 1]
                                        group_corrs.append({
                                            'Group': str(group),
                                            'Correlation': group_corr,
                                            'Sample Size': len(group_df)
                                        })
                                
                                if group_corrs:
                                    st.dataframe(pd.DataFrame(group_corrs))
            else:
                st.warning("Please select at least 2 columns for correlation analysis.")
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
    
    # Tab 2: Dimensionality Reduction
    with tab2:
        st.subheader("Dimensionality Reduction")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
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
                    
                    # Perform PCA
                    try:
                        pca_results, loadings, explained_var = perform_pca(
                            df[selected_cols], 
                            n_components=n_components,
                            standardize=standardize
                        )
                        
                        # Plot first two components
                        fig = px.scatter(
                            pca_results, 
                            x='PC1', 
                            y='PC2',
                            title=f"PCA: First two components explain {100 * sum(explained_var[:2]):.1f}% of variance",
                            color=df[color_by] if color_by else None
                        )
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
                                fig = px.scatter(
                                    tsne_results, 
                                    x='TSNE1', 
                                    y='TSNE2',
                                    title="t-SNE Projection",
                                    color=df[color_by] if color_by else None
                                )
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
                                fig = px.scatter(
                                    umap_results, 
                                    x='UMAP1', 
                                    y='UMAP2',
                                    title="UMAP Projection",
                                    color=df[color_by] if color_by else None
                                )
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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
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
                    
                    # Perform K-Means
                    if st.button("Run K-Means"):
                        try:
                            cluster_assignments, kmeans_model = perform_kmeans(
                                df[selected_cols],
                                n_clusters=n_clusters
                            )
                            
                            # Add cluster assignments to dataframe for visualization
                            vis_df = df[selected_cols].copy()
                            vis_df['Cluster'] = cluster_assignments['Cluster']
                            
                            # Perform PCA for visualization if needed
                            if len(selected_cols) > 2:
                                pca_results, _, _ = perform_pca(df[selected_cols])
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
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show cluster centers if PCA was used
                            if len(selected_cols) > 2:
                                # Transform cluster centers to PCA space
                                centers_pca = StandardScaler().fit_transform(kmeans_model.cluster_centers_)
                                pca = PCA(n_components=2)
                                pca.fit(df[selected_cols])
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
                                            size=12,
                                            line=dict(width=2)
                                        ),
                                        name='Cluster Centers'
                                    )
                                )
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
                    
                    # Perform DBSCAN
                    if st.button("Run DBSCAN"):
                        try:
                            cluster_assignments = perform_dbscan(
                                df[selected_cols],
                                eps=eps,
                                min_samples=min_samples
                            )
                            
                            # Add cluster assignments to dataframe for visualization
                            vis_df = df[selected_cols].copy()
                            vis_df['Cluster'] = cluster_assignments['Cluster']
                            
                            # Perform PCA for visualization if needed
                            if len(selected_cols) > 2:
                                pca_results, _, _ = perform_pca(df[selected_cols])
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
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
        
        if numeric_cols and categorical_cols:
            # Select test type
            test_type = st.radio(
                "Select test type:",
                ["T-Test (compare two groups)", "ANOVA (compare multiple groups)"],
                horizontal=True
            )
            
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
            
            if test_type == "T-Test (compare two groups)":
                # Perform T-Test
                if st.button("Run T-Test"):
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
            
            elif test_type == "ANOVA (compare multiple groups)":
                # Perform ANOVA
                if st.button("Run ANOVA"):
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
        else:
            st.warning("Need both numeric and categorical columns for hypothesis testing.")
