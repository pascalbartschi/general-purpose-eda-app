import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from pathlib import Path

def app():
    st.header("Exploratory Data Analysis")
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.warning("Please load data in the 'Data Loading and Filtering' tab first.")
        return
    
    # Use filtered data if available, otherwise use original data
    if 'filtered_data' in st.session_state:
        df = st.session_state['filtered_data']
        st.info("Using filtered data for analysis")
    else:
        df = st.session_state['data']
        st.info("Using original data for analysis")
    
    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(f"Dataset shape: {df.shape}")
    
    # Data summary
    with st.expander("Data Summary"):
        st.write("Data types:")
        st.write(df.dtypes)
        
        st.write("Summary statistics:")
        st.write(df.describe())
        
        st.write("Missing values:")
        missing = df.isnull().sum()
        st.write(missing[missing > 0])
    
    # Data visualization options
    st.subheader("Data Visualization")
    
    viz_type = st.selectbox(
        "Select Visualization Type",
        ["Distribution Analysis", "Correlation Analysis", "Time Series Analysis", "Categorical Analysis"]
    )
    
    if viz_type == "Distribution Analysis":
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select column for distribution analysis", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Histogram of {selected_col}")
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
            
            with col2:
                st.subheader(f"Box Plot of {selected_col}")
                fig, ax = plt.subplots()
                sns.boxplot(y=df[selected_col].dropna(), ax=ax)
                st.pyplot(fig)
                
            # Statistics
            st.subheader(f"Statistics for {selected_col}")
            stats_df = pd.DataFrame({
                'Mean': [df[selected_col].mean()],
                'Median': [df[selected_col].median()],
                'Std Dev': [df[selected_col].std()],
                'Min': [df[selected_col].min()],
                'Max': [df[selected_col].max()]
            })
            st.write(stats_df)
        else:
            st.warning("No numeric columns available for distribution analysis")
    
    elif viz_type == "Correlation Analysis":
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if len(numeric_cols) > 1:
            st.subheader("Correlation Matrix")
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", 
                        cmap='coolwarm', ax=ax, linewidths=0.5)
            st.pyplot(fig)
            
            # Scatter plot for selected features
            st.subheader("Scatter Plot")
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Select X axis", numeric_cols)
            with col2:
                remaining_cols = [col for col in numeric_cols if col != x_col]
                y_col = st.selectbox("Select Y axis", remaining_cols if remaining_cols else numeric_cols)
            
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least two numeric columns for correlation analysis")
    
    elif viz_type == "Time Series Analysis":
        # Check if any column might be a date
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    pd.to_datetime(df[col])
                    date_cols.append(col)
                except:
                    pass
        
        if date_cols:
            selected_date_col = st.selectbox("Select date column", date_cols)
            
            # Convert to datetime
            df_ts = df.copy()
            df_ts[selected_date_col] = pd.to_datetime(df_ts[selected_date_col])
            
            # Select value column to plot
            value_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if value_cols:
                selected_value_col = st.selectbox("Select value to plot", value_cols)
                
                # Resample options
                resample_options = {
                    "No resampling": None,
                    "Daily": 'D',
                    "Weekly": 'W',
                    "Monthly": 'M',
                    "Quarterly": 'Q',
                    "Yearly": 'Y'
                }
                
                resample_choice = st.selectbox("Resample time series", list(resample_options.keys()))
                
                df_ts = df_ts.sort_values(by=selected_date_col)
                
                if resample_options[resample_choice]:
                    # Set date as index for resampling
                    df_ts = df_ts.set_index(selected_date_col)
                    resampled = df_ts[selected_value_col].resample(resample_options[resample_choice]).mean()
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    resampled.plot(ax=ax)
                    ax.set_title(f"{selected_value_col} ({resample_choice} Average)")
                    ax.set_ylabel(selected_value_col)
                    st.pyplot(fig)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    plt.plot(df_ts[selected_date_col], df_ts[selected_value_col])
                    ax.set_title(f"{selected_value_col} over time")
                    ax.set_xlabel(selected_date_col)
                    ax.set_ylabel(selected_value_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("No numeric columns available for time series analysis")
        else:
            st.warning("No date columns detected. Please ensure your data contains a column with date values.")
            
    elif viz_type == "Categorical Analysis":
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if cat_cols:
            selected_cat_col = st.selectbox("Select categorical column", cat_cols)
            
            # Count plot
            st.subheader(f"Count plot for {selected_cat_col}")
            
            # Get value counts and limit to top categories if there are too many
            value_counts = df[selected_cat_col].value_counts()
            if len(value_counts) > 15:
                top_n = st.slider("Select number of top categories to display", 5, 30, 10)
                value_counts = value_counts.head(top_n)
                st.info(f"Showing top {top_n} categories out of {len(df[selected_cat_col].unique())}")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Pie chart
            st.subheader(f"Percentage distribution for {selected_cat_col}")
            fig, ax = plt.subplots(figsize=(10, 6))
            value_counts.plot.pie(autopct='%1.1f%%', ax=ax)
            st.pyplot(fig)
            
            # Cross-tabulation with another categorical column
            other_cat_cols = [col for col in cat_cols if col != selected_cat_col]
            if other_cat_cols:
                st.subheader("Cross-tabulation")
                other_col = st.selectbox("Select another categorical column", other_cat_cols)
                
                # Create crosstab
                cross_tab = pd.crosstab(df[selected_cat_col], df[other_col])
                st.write(cross_tab)
                
                # Visualize crosstab as heatmap
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(cross_tab, annot=True, fmt='d', cmap='viridis', ax=ax)
                st.pyplot(fig)
        else:
            st.warning("No categorical columns available for analysis")
    
    # Save plots
    st.subheader("Download Visualizations")
    st.write("To save a visualization, right-click on it and select 'Save image as...'")
