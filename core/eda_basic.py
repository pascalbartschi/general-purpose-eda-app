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
            # Check if it's boolean-like
            if set(df[col].dropna().unique()).issubset({0, 1, True, False}):
                type_categories['Boolean'].append(col)
            else:
            # Check if it's integer-like
                type_categories['Numeric'].append(col)
                
        elif pd.api.types.is_categorical_dtype(df[col]):
            # Check if it's a boolean-like categorical
            type_categories['Categorical'].append(col)
            
        elif pd.api.types.is_datetime64_dtype(df[col]):
            # Check if it's a datetime-like column
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

def plot_scatter(df, x_col, y_col, color_col=None):
    title = f"Scatter Plot: {y_col} vs {x_col}"
    if color_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title)
    else:
        fig = px.scatter(df, x=x_col, y=y_col, title=title)

    try:
        x_vals = df[x_col].dropna()
        y_vals = df[y_col].dropna()
        common = df[[x_col, y_col]].dropna()
        fit_type = st.selectbox("Select fit type:", ["Linear", "Quadratic", "Cubic", "Exponential", "Logarithmic", "Polynomial (n-order)"], index=0)

        if fit_type == "Linear":
            degree = 1
        elif fit_type == "Quadratic":
            degree = 2
        elif fit_type == "Cubic":
            degree = 3
        elif fit_type == "Exponential":
            coeffs = np.polyfit(common[x_col], np.log(common[y_col] + 1e-5), 1)
            y_fit = np.exp(coeffs[1] + coeffs[0] * common[x_col])
            fig.add_trace(go.Scatter(x=common[x_col], y=y_fit, mode='lines', name='Exponential Fit'))
            corr = np.corrcoef(common[y_col], y_fit)[0, 1]
            fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text=f"Exp Corr: {corr:.2f}",
                               showarrow=False, bgcolor="white")
            return fig
        elif fit_type == "Logarithmic":
            log_x = np.log(common[x_col] + 1e-5)
            coeffs = np.polyfit(log_x, common[y_col], 1)
            y_fit = coeffs[0] * np.log(common[x_col] + 1e-5) + coeffs[1]
            fig.add_trace(go.Scatter(x=common[x_col], y=y_fit, mode='lines', name='Log Fit'))
            corr = np.corrcoef(common[y_col], y_fit)[0, 1]
            fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text=f"Log Corr: {corr:.2f}",
                               showarrow=False, bgcolor="white")
            return fig
        elif fit_type == "Polynomial (n-order)":
            degree = st.slider("Select polynomial degree:", min_value=1, max_value=10, value=4)
        else:
            degree = 1

        coeffs = np.polyfit(common[x_col], common[y_col], degree)
        poly = np.poly1d(coeffs)
        x_sorted = np.sort(common[x_col])
        y_fit = poly(x_sorted)
        fig.add_trace(go.Scatter(x=x_sorted, y=y_fit, mode='lines', name=f'{fit_type} Fit'))
        corr = np.corrcoef(common[y_col], poly(common[x_col]))[0, 1]
        fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text=f"Corr: {corr:.2f}",
                           showarrow=False, bgcolor="white")
    except Exception as e:
        st.warning(f"Could not compute fit: {e}")

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

        df_temp = df.copy()

        x_col = st.selectbox("Select X-axis (time-like or numeric):", options=df_temp.columns)
        
        if pd.api.types.is_datetime64_any_dtype(df_temp[x_col]):
            df_temp['x_val'] = df_temp[x_col]
            x_label = x_col
        else:
            try:
                df_temp['x_val'] = pd.to_numeric(df_temp[x_col])
                x_unit = st.text_input(f"Enter unit for x-axis [{x_col}]:", value="units")
                x_label = f"{x_col} [{x_unit}]"
            except:
                st.error("Selected x-axis column must be datetime or numeric.")
                st.stop()

        numeric_cols = df_temp.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            st.warning("No numeric columns available for Y-axis.")
            st.stop()

        y_col = st.selectbox("Select Y-axis (numeric):", options=numeric_cols)
        df_temp['y_val'] = df_temp[y_col]

        # Optional grouping by category
        st.subheader("Grouping Options")
        categorical_cols = df_temp.select_dtypes(exclude=['number', 'datetime64']).columns.tolist()
        # Also include columns that might be categorical but stored as object/string
        for col in df_temp.columns:
            if col not in categorical_cols and df_temp[col].dtype == 'object':
                if df_temp[col].nunique() <= 50:  # Reasonable number of categories
                    categorical_cols.append(col)
        
        group_col = None
        if categorical_cols:
            use_grouping = st.checkbox("Group by category column?", value=False)
            if use_grouping:
                group_col = st.selectbox("Select column for grouping:", options=categorical_cols)
                
                # Option to select specific categories
                if group_col:
                    unique_categories = df_temp[group_col].unique()
                    st.write(f"Found {len(unique_categories)} unique categories in '{group_col}'")
                    
                    if len(unique_categories) > 20:
                        st.warning(f"Large number of categories ({len(unique_categories)}). Consider selecting specific ones below.")
                    
                    # Allow user to select specific categories
                    selected_categories = st.multiselect(
                        f"Select specific categories from '{group_col}' (leave empty for all):",
                        options=unique_categories,
                        default=unique_categories[:min(10, len(unique_categories))]  # Default to first 10
                    )
                    
                    if selected_categories:
                        df_temp = df_temp[df_temp[group_col].isin(selected_categories)]

        # Prepare data for plotting
        if group_col:
            # Keep the grouping column along with x and y values
            df_temp = df_temp[['x_val', 'y_val', group_col]].dropna().sort_values('x_val')
            
            # Plotting with grouping
            fig = px.line(df_temp, x='x_val', y='y_val', color=group_col,
                        title=f"{y_col} vs {x_col} (grouped by {group_col})",
                        labels={'x_val': x_label, 'y_val': y_col, group_col: group_col})
        else:
            # Drop missing values
            df_temp = df_temp[['x_val', 'y_val']].dropna().sort_values('x_val')
            
            # Plotting without grouping
            fig = px.line(df_temp, x='x_val', y='y_val', title=f"{y_col} vs {x_col}",
                        labels={'x_val': x_label, 'y_val': y_col})
        
        st.plotly_chart(fig, use_container_width=True)

        # --- Basic Time Series Stats ---
        st.subheader("Time Series Statistics")
        st.write(f"**{y_col} Summary Stats:**")
        st.write(df_temp['y_val'].describe())

        # Optional rolling mean
        window = st.slider("Rolling Mean Window (points):", min_value=1, max_value=min(100, len(df_temp)), value=5)
        
        if group_col:
            # Calculate rolling mean for each group
            df_temp['rolling_mean'] = df_temp.groupby(group_col)['y_val'].rolling(window=window).mean().reset_index(0, drop=True)
            
            fig2 = px.line(df_temp, x='x_val', y='rolling_mean', color=group_col,
                        title=f"{y_col} - Rolling Mean ({window} points) grouped by {group_col}",
                        labels={'x_val': x_label, 'rolling_mean': f"{y_col} (Rolling Mean)", group_col: group_col})
        else:
            df_temp['rolling_mean'] = df_temp['y_val'].rolling(window=window).mean()
            
            fig2 = px.line(df_temp, x='x_val', y='rolling_mean', title=f"{y_col} - Rolling Mean ({window} points)",
                        labels={'x_val': x_label, 'rolling_mean': f"{y_col} (Rolling Mean)"})
        
        st.plotly_chart(fig2, use_container_width=True)

        # Optional linear trend line
        st.subheader("Add Linear Trend Line?")
        if st.checkbox("Show Linear Trend"):
            coeffs = np.polyfit(df_temp['x_val'], df_temp['y_val'], 1)
            trend = np.poly1d(coeffs)
            df_temp['trend'] = trend(df_temp['x_val'])

            fig3 = px.line(df_temp, x='x_val', y='trend', title=f"{y_col} - Linear Trend",
                        labels={'x_val': x_label, 'trend': 'Trend'})
            st.plotly_chart(fig3, use_container_width=True)
