"""
Module for basic exploratory data analysis functions of the EDA app.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def get_data_types_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Get a summary of column types in the dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict[str, List[str]]: Dictionary with data type categories as keys and column names as values
    """
    # Initialize type categories
    type_categories = {
        'Numeric': [],
        'Categorical': [],
        'Boolean': [],
        'DateTime': [],
        'Text': [],
        'Other': []
    }
    
    # Categorize each column
    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(df[col]):
            if set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                type_categories['Boolean'].append(col)
            else:
                type_categories['Numeric'].append(col)
                
        elif pd.api.types.is_categorical_dtype(df[col]):
            type_categories['Categorical'].append(col)
            
        elif pd.api.types.is_datetime64_dtype(df[col]):
            type_categories['DateTime'].append(col)
            
        elif pd.api.types.is_string_dtype(df[col]) or df[col].dtype == 'object':
            # Check if it's likely text (more than N unique values)
            if df[col].nunique() > 0.5 * len(df) and df[col].nunique() > 20:
                type_categories['Text'].append(col)
            else:
                type_categories['Categorical'].append(col)
                
        else:
            type_categories['Other'].append(col)
    
    # Remove empty categories
    return {k: v for k, v in type_categories.items() if v}

def plot_numeric_distribution(df: pd.DataFrame, 
                             column: str,
                             log_scale: bool = False,
                             show_mean: bool = True,
                             show_median: bool = True,
                             bins: int = 30) -> go.Figure:
    """
    Create a histogram with KDE for a numeric column.
    
    Args:
        df: Input dataframe
        column: Column name to visualize
        log_scale: Whether to use log scale
        show_mean: Whether to show mean line
        show_median: Whether to show median line
        bins: Number of histogram bins
        
    Returns:
        go.Figure: Plotly figure object
    """
    fig = px.histogram(
        df, 
        x=column, 
        nbins=bins,
        histnorm='density',
        title=f"Distribution of {column}",
        labels={column: column},
        marginal="box"
    )
    
    # Add KDE
    try:
        import scipy.stats as stats
        kde = stats.gaussian_kde(df[column].dropna())
        x_range = np.linspace(df[column].min(), df[column].max(), 1000)
        y_kde = kde(x_range)
        
        fig.add_trace(
            go.Scatter(
                x=x_range, 
                y=y_kde, 
                mode='lines', 
                name='KDE',
                line=dict(color='red')
            )
        )
    except:
        pass
    
    # Add mean and median lines
    if show_mean:
        mean_val = df[column].mean()
        fig.add_vline(x=mean_val, 
                      line_dash="dash", 
                      line_color="green",
                      annotation_text=f"Mean: {mean_val:.2f}")
    
    if show_median:
        median_val = df[column].median()
        fig.add_vline(x=median_val, 
                      line_dash="dash", 
                      line_color="blue",
                      annotation_text=f"Median: {median_val:.2f}")
    
    # Set log scale if requested
    if log_scale:
        fig.update_layout(xaxis_type="log")
    
    return fig

def plot_categorical_distribution(df: pd.DataFrame, column: str) -> go.Figure:
    """
    Create a bar plot for a categorical column.
    
    Args:
        df: Input dataframe
        column: Column name to visualize
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Get value counts
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'count']
    
    # Sort by count, descending
    value_counts = value_counts.sort_values('count', ascending=False)
    
    # Create a bar plot
    fig = px.bar(
        value_counts,
        x=column,
        y='count',
        title=f"Distribution of {column}",
        labels={column: column, 'count': 'Count'},
        text='count'
    )
    
    # Update layout for better readability
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
        xaxis=dict(
            type='category',
            categoryorder='total descending'
        )
    )
    
    # Handle large number of categories
    if value_counts.shape[0] > 10:
        fig.update_layout(
            xaxis=dict(
                tickangle=45,
                tickmode='auto',
                tickfont=dict(size=10),
            )
        )
    
    return fig

def plot_datetime_distribution(df: pd.DataFrame, column: str, freq: str = 'M') -> go.Figure:
    """
    Create a line plot for a datetime column.
    
    Args:
        df: Input dataframe
        column: Column name to visualize
        freq: Frequency for grouping ('D' for daily, 'W' for weekly, 'M' for monthly, 'Y' for yearly)
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Make a copy to avoid SettingWithCopyWarning
    temp_df = df.copy()
    
    # Make sure the column is datetime
    temp_df[column] = pd.to_datetime(temp_df[column], errors='coerce')
    
    # Group by frequency
    if freq == 'D':
        temp_df['period'] = temp_df[column].dt.date
    elif freq == 'W':
        temp_df['period'] = temp_df[column].dt.isocalendar().week
    elif freq == 'M':
        temp_df['period'] = temp_df[column].dt.to_period('M').astype(str)
    elif freq == 'Y':
        temp_df['period'] = temp_df[column].dt.year
    
    # Count by period
    time_counts = temp_df['period'].value_counts().reset_index()
    time_counts.columns = ['period', 'count']
    time_counts = time_counts.sort_values('period')
    
    # Create plot
    fig = px.line(
        time_counts, 
        x='period', 
        y='count',
        title=f"Distribution of {column} by {freq}",
        markers=True
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title="Period",
        yaxis_title="Count"
    )
    
    # Handle large number of periods
    if time_counts.shape[0] > 20:
        fig.update_layout(
            xaxis=dict(
                tickangle=45,
                tickmode='auto', 
                nticks=20
            )
        )
    
    return fig

def get_descriptive_stats(df: pd.DataFrame, 
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate descriptive statistics for numerical columns.
    
    Args:
        df: Input dataframe
        columns: Specific columns to analyze (None for all numeric)
        
    Returns:
        pd.DataFrame: Dataframe with descriptive statistics
    """
    # Select columns to analyze
    if columns is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    else:
        numeric_cols = [col for col in columns if pd.api.types.is_numeric_dtype(df[col])]
    
    if not numeric_cols:
        return pd.DataFrame()
    
    # Calculate statistics
    stats = df[numeric_cols].describe().T
    
    # Add more statistics
    stats['variance'] = df[numeric_cols].var()
    stats['skewness'] = df[numeric_cols].skew()
    stats['kurtosis'] = df[numeric_cols].kurtosis()
    stats['missing'] = df[numeric_cols].isna().sum()
    stats['missing_pct'] = 100 * df[numeric_cols].isna().mean()
    
    # Calculate IQR (Interquartile Range)
    stats['iqr'] = stats['75%'] - stats['25%']
    
    # Calculate range
    stats['range'] = stats['max'] - stats['min']
    
    # Calculate coefficient of variation (CV)
    stats['cv'] = 100 * stats['std'] / stats['mean']
    
    # Round values for display
    stats = stats.round(2)
    
    return stats

def plot_boxplot(df: pd.DataFrame, 
                numeric_col: str, 
                group_col: Optional[str] = None) -> go.Figure:
    """
    Create a boxplot for a numerical feature, optionally grouped by category.
    
    Args:
        df: Input dataframe
        numeric_col: Numeric column to visualize
        group_col: Optional categorical column for grouping
        
    Returns:
        go.Figure: Plotly figure object
    """
    title = f"Boxplot of {numeric_col}"
    
    if group_col:
        title += f" grouped by {group_col}"
        fig = px.box(
            df,
            x=group_col,
            y=numeric_col,
            title=title,
            labels={numeric_col: numeric_col, group_col: group_col},
            points="outliers"  # Only show outliers as individual points
        )
    else:
        fig = px.box(
            df,
            y=numeric_col,
            title=title,
            labels={numeric_col: numeric_col},
            points="outliers"  # Only show outliers as individual points
        )
        
    return fig

def plot_scatter(df: pd.DataFrame, 
                x_col: str, 
                y_col: str, 
                color_col: Optional[str] = None) -> go.Figure:
    """
    Create a scatter plot between two numerical features.
    
    Args:
        df: Input dataframe
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        color_col: Optional column for coloring points
        
    Returns:
        go.Figure: Plotly figure object
    """
    title = f"Scatter Plot: {y_col} vs {x_col}"
    
    if color_col:
        title += f" (colored by {color_col})"
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            labels={x_col: x_col, y_col: y_col, color_col: color_col}
        )
    else:
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            title=title,
            labels={x_col: x_col, y_col: y_col}
        )
    
    # Add trend line
    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
        try:
            fig.update_layout(
                shapes=[
                    dict(
                        type='line',
                        yref='y',
                        xref='x',
                        x0=df[x_col].min(),
                        y0=np.polyval(np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1), df[x_col].min()),
                        x1=df[x_col].max(),
                        y1=np.polyval(np.polyfit(df[x_col].dropna(), df[y_col].dropna(), 1), df[x_col].max()),
                        line=dict(
                            color="red",
                            width=2,
                            dash="dash",
                        )
                    )
                ]
            )
            
            # Calculate correlation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            fig.add_annotation(
                x=0.95,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Correlation: {corr:.2f}",
                showarrow=False,
                font=dict(
                    size=12,
                    color="black"
                ),
                bordercolor="black",
                bgcolor="white",
                borderwidth=1,
                borderpad=4
            )
        except:
            # Skip trend line if there's an error
            pass
    
    return fig

def basic_eda_ui(df: pd.DataFrame) -> None:
    """
    Streamlit UI for basic exploratory data analysis.
    
    Args:
        df: Input dataframe
    """
    st.title("Basic Exploratory Data Analysis")
    
    if df is None or df.empty:
        st.warning("Please upload and clean a dataset first.")
        return
    
    # Data Type Summary
    st.subheader("Data Types Summary")
    type_summary = get_data_types_summary(df)
    
    for type_name, columns in type_summary.items():
        st.write(f"**{type_name}** ({len(columns)}): {', '.join(columns)}")
    
    # Tabs for different analysis types
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Feature Distributions", 
        "Descriptive Statistics", 
        "Outlier Analysis", 
        "Feature Relationships", 
        "Time Series Analysis"
    ])
    
    # Tab 1: Feature Distributions
    with tab1:
        st.subheader("Feature Distribution Analysis")
        
        # Select column to visualize
        column = st.selectbox(
            "Select column to visualize:",
            options=df.columns.tolist()
        )
        
        # Distribution plot based on data type
        if column:
            if pd.api.types.is_numeric_dtype(df[column]):
                # Options for numeric distribution
                col1, col2 = st.columns(2)
                with col1:
                    log_scale = st.checkbox("Use log scale", value=False)
                with col2:
                    bins = st.slider("Number of bins:", 5, 100, 30)
                
                show_mean = st.checkbox("Show mean line", value=True)
                show_median = st.checkbox("Show median line", value=True)
                
                # Create and show plot
                fig = plot_numeric_distribution(
                    df, column, log_scale, show_mean, show_median, bins
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display summary statistics
                st.write("**Summary Statistics:**")
                stats = df[column].describe()
                st.write(pd.DataFrame(stats).T)
                
            elif pd.api.types.is_datetime64_dtype(df[column]):
                # Options for datetime distribution
                time_freq = st.selectbox(
                    "Select time aggregation level:",
                    options=["Day", "Week", "Month", "Year"],
                    index=2  # Default to Month
                )
                
                # Map selection to frequency code
                freq_map = {"Day": "D", "Week": "W", "Month": "M", "Year": "Y"}
                selected_freq = freq_map[time_freq]
                
                # Create and show plot
                fig = plot_datetime_distribution(df, column, selected_freq)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # For categorical data
                # Create and show plot
                fig = plot_categorical_distribution(df, column)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display value counts
                st.write("**Value Counts:**")
                value_counts = df[column].value_counts()
                st.write(pd.DataFrame(value_counts).head(20))
                
                if len(value_counts) > 20:
                    st.info(f"Showing top 20 values out of {len(value_counts)} unique values.")
    
    # Tab 2: Descriptive Statistics
    with tab2:
        st.subheader("Descriptive Statistics")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Options for statistics
            selected_cols = st.multiselect(
                "Select columns for statistics (defaults to all numeric):",
                options=df.columns.tolist(),
                default=numeric_cols
            )
            
            # Calculate and show statistics
            stats = get_descriptive_stats(df, selected_cols)
            
            if not stats.empty:
                st.write("**Detailed Statistics:**")
                st.dataframe(stats)
                
                # Option to download statistics
                if st.button("Download Statistics as CSV"):
                    # In a real app, you'd implement the download functionality here
                    st.write("Download functionality would be implemented here.")
            else:
                st.warning("No numeric columns selected for statistics.")
        else:
            st.warning("No numeric columns in dataset for statistics.")
    
    # Tab 3: Outlier Analysis
    with tab3:
        st.subheader("Outlier Analysis")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Select column and grouping
            col1, col2 = st.columns(2)
            
            with col1:
                numeric_col = st.selectbox(
                    "Select numeric column for analysis:",
                    options=numeric_cols
                )
            
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            categorical_cols.insert(0, None)  # Add None as first option
            
            with col2:
                group_col = st.selectbox(
                    "Group by (optional):",
                    options=categorical_cols
                )
            
            if numeric_col:
                # Create and show boxplot
                fig = plot_boxplot(df, numeric_col, group_col)
                st.plotly_chart(fig, use_container_width=True)
                
                # Outlier detection options
                st.subheader("Find Outliers")
                
                detect_method = st.radio(
                    "Select outlier detection method:",
                    ["IQR Method", "Z-Score Method"],
                    horizontal=True
                )
                
                if detect_method == "IQR Method":
                    factor = st.slider("IQR factor (typically 1.5):", 0.5, 3.0, 1.5, 0.1)
                    
                    Q1 = df[numeric_col].quantile(0.25)
                    Q3 = df[numeric_col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers = df[(df[numeric_col] < lower_bound) | (df[numeric_col] > upper_bound)]
                    
                    st.write(f"**Outliers (IQR method, factor={factor}):**")
                    st.write(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
                    st.write(f"Found {len(outliers)} outliers out of {len(df)} records ({100*len(outliers)/len(df):.2f}%).")
                
                else:  # Z-Score Method
                    z_threshold = st.slider("Z-Score threshold:", 1.0, 5.0, 3.0, 0.1)
                    
                    mean = df[numeric_col].mean()
                    std = df[numeric_col].std()
                    
                    z_scores = (df[numeric_col] - mean) / std
                    outliers = df[abs(z_scores) > z_threshold]
                    
                    st.write(f"**Outliers (Z-score method, threshold={z_threshold}):**")
                    st.write(f"Found {len(outliers)} outliers out of {len(df)} records ({100*len(outliers)/len(df):.2f}%).")
                
                if not outliers.empty and len(outliers) < 100:
                    st.dataframe(outliers)
                elif len(outliers) >= 100:
                    st.write("Too many outliers to display. Here's a sample:")
                    st.dataframe(outliers.sample(100))
                else:
                    st.write("No outliers detected.")
        else:
            st.warning("No numeric columns in dataset for outlier analysis.")
    
    # Tab 4: Feature Relationships
    with tab4:
        st.subheader("Feature Relationships")
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox(
                    "Select X-axis column:",
                    options=df.columns.tolist()
                )
            
            with col2:
                y_col = st.selectbox(
                    "Select Y-axis column:",
                    options=[c for c in df.columns if c != x_col]
                )
            
            # Get all columns for color option
            all_cols = df.columns.tolist()
            all_cols.insert(0, None)  # Add None as first option
            
            with col3:
                color_col = st.selectbox(
                    "Color by (optional):",
                    options=all_cols
                )
            
            # Create and show scatter plot
            fig = plot_scatter(df, x_col, y_col, color_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation matrix section
            st.subheader("Correlation Matrix")
            
            if len(numeric_cols) > 0:
                selected_cols = st.multiselect(
                    "Select columns for correlation matrix:",
                    options=numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))]
                )
                
                if selected_cols:
                    correlation_method = st.radio(
                        "Correlation method:",
                        ["Pearson", "Spearman", "Kendall"],
                        horizontal=True
                    )
                    
                    corr_matrix = df[selected_cols].corr(method=correlation_method.lower())
                    
                    # Create heatmap
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=".2f",
                        color_continuous_scale="RdBu_r",
                        title=f"{correlation_method} Correlation Matrix"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation matrix as table
                    st.write("**Correlation Matrix (Table):**")
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm'))
        else:
            st.warning("Need at least 2 numeric columns for relationship analysis.")
    
    # Tab 5: Time Series Analysis
    with tab5:
        st.subheader("Time Series Analysis")
        
        # Detect datetime columns
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Also check if any object columns might be convertible to datetime
        for col in df.select_dtypes(include=['object']):
            try:
                # Try to convert a sample
                sample = pd.to_datetime(df[col].dropna().iloc[0])
                if isinstance(sample, pd.Timestamp):
                    datetime_cols.append(col)
            except:
                pass
        
        if datetime_cols:
            datetime_col = st.selectbox(
                "Select datetime column:",
                options=datetime_cols
            )
            
            if datetime_col:
                # Ensure column is datetime type
                df_temp = df.copy()
                df_temp[datetime_col] = pd.to_datetime(df_temp[datetime_col], errors='coerce')
                
                # Select numeric column to analyze
                numeric_cols = df_temp.select_dtypes(include=['number']).columns.tolist()
                
                if numeric_cols:
                    numeric_col = st.selectbox(
                        "Select numeric column for time series analysis:",
                        options=numeric_cols
                    )
                    
                    # Group data by time periods
                    time_freq = st.selectbox(
                        "Select time aggregation level:",
                        options=["Day", "Week", "Month", "Year"],
                        index=2  # Default to Month
                    )
                    
                    # Map selection to pandas frequency code
                    freq_map = {
                        "Day": "D",
                        "Week": "W",
                        "Month": "M",
                        "Year": "Y"
                    }
                    selected_freq = freq_map[time_freq]
                    
                    # Aggregate data
                    if selected_freq == "D":
                        df_temp['period'] = df_temp[datetime_col].dt.date
                    elif selected_freq == "W":
                        df_temp['period'] = df_temp[datetime_col].dt.to_period("W").dt.start_time
                    elif selected_freq == "M":
                        df_temp['period'] = df_temp[datetime_col].dt.to_period("M").dt.start_time
                    else:  # yearly
                        df_temp['period'] = df_temp[datetime_col].dt.to_period("Y").dt.start_time
                    
                    # Calculate aggregations
                    agg_data = df_temp.groupby('period')[numeric_col].agg(['mean', 'median', 'min', 'max', 'count']).reset_index()
                    
                    # Plot time series
                    fig = px.line(
                        agg_data, 
                        x='period', 
                        y=['mean', 'median', 'min', 'max'],
                        title=f"Time Series Analysis of {numeric_col} by {time_freq}",
                        labels={'value': numeric_col, 'period': 'Time Period', 'variable': 'Metric'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show data table
                    st.write("**Time Series Aggregated Data:**")
                    st.dataframe(agg_data)
                else:
                    st.warning("No numeric columns available for time series analysis.")
        else:
            st.warning("No datetime columns detected in the dataset.")
