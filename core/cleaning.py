"""
Module for data cleaning and transformation functions of the EDA app.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import streamlit as st

def detect_non_standard_missing(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """
    Detect non-standard missing values in dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict[str, List[Any]]: Dictionary with columns as keys and lists of detected non-standard nulls
    """
    potential_nulls = ['', ' ', 'nan', 'null', 'none', 'na', 'n/a', '-', '--', '?', 'unknown', 'undefined', 'missing']
    
    non_standard_nulls = {}
    
    for col in df.columns:
        # Convert to string to handle all types
        col_vals = df[col].astype(str).str.lower()
        
        # Find values that match potential null patterns
        found_nulls = [val for val in df[col].unique() 
                      if isinstance(val, str) and val.lower() in potential_nulls]
        
        if found_nulls:
            non_standard_nulls[col] = found_nulls
    
    return non_standard_nulls

def replace_non_standard_missing(df: pd.DataFrame, 
                               null_values_dict: Dict[str, List[Any]]) -> pd.DataFrame:
    """
    Replace non-standard missing values with NaN.
    
    Args:
        df: Input dataframe
        null_values_dict: Dictionary with columns as keys and lists of values to replace
        
    Returns:
        pd.DataFrame: Dataframe with non-standard nulls replaced with NaN
    """
    df_copy = df.copy()
    
    for col, values in null_values_dict.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].replace(values, np.nan)
    
    return df_copy

def handle_missing_values(df: pd.DataFrame, 
                         method: str = 'leave', 
                         fill_value: Optional[Any] = None,
                         n_neighbors: int = 5) -> pd.DataFrame:
    """
    Handle missing values in the dataframe.
    
    Args:
        df: Input dataframe
        method: Method to handle missing values ('leave', 'fill', 'knn')
        fill_value: Value to use if method is 'fill'
        n_neighbors: Number of neighbors for KNN imputation
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    df_copy = df.copy()
    
    if method == 'leave':
        # Leave as NaN
        return df_copy
    
    elif method == 'fill':
        # Fill with specific value
        return df_copy.fillna(fill_value)
    
    elif method == 'knn':
        # KNN imputation for numeric columns
        numeric_cols = df_copy.select_dtypes(include=['number']).columns.tolist()
        
        if numeric_cols:
            # Extract numeric data for KNN imputation
            numeric_data = df_copy[numeric_cols]
            
            # Use KNN imputer
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed_data = imputer.fit_transform(numeric_data)
            
            # Replace the imputed data in the dataframe
            df_copy[numeric_cols] = imputed_data
        
        # For non-numeric columns, use mode imputation
        for col in df_copy.select_dtypes(exclude=['number']).columns:
            if df_copy[col].isna().any():
                mode_val = df_copy[col].mode()[0]
                df_copy[col] = df_copy[col].fillna(mode_val)
                
        return df_copy
    else:
        raise ValueError(f"Unsupported method: {method}")

def convert_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert and infer appropriate data types for columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with inferred data types
    """
    df_copy = df.copy()
    
    # Attempt to convert to numeric where appropriate
    for col in df_copy.columns:
        # Try converting to numeric
        if df_copy[col].dtype == 'object':
            try:
                # Check if values can be converted to numeric
                pd.to_numeric(df_copy[col], errors='coerce')
                # If there aren't too many NaN after conversion, apply it
                if df_copy[col].isna().mean() < 0.3:  # Less than 30% NaNs
                    df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            except:
                pass
                
            # Try datetime conversion
            try:
                pd.to_datetime(df_copy[col], errors='coerce')
                # If there aren't too many NaT after conversion, apply it
                if df_copy[col].isna().mean() < 0.3:  # Less than 30% NaNs
                    df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
            except:
                pass
    
    # Convert categorical columns
    for col in df_copy.select_dtypes(include=['object']).columns:
        # If column has low cardinality relative to dataset, convert to categorical
        if df_copy[col].nunique() / len(df_copy) < 0.05:  # Less than 5% unique values
            df_copy[col] = df_copy[col].astype('category')
    
    return df_copy

def normalize_column(series: pd.Series, method: str = 'minmax') -> pd.Series:
    """
    Normalize a numerical series.
    
    Args:
        series: Input series
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        pd.Series: Normalized series
    """
    # Handle non-numeric data
    if not pd.api.types.is_numeric_dtype(series):
        return series
    
    # Extract values and reshape for sklearn
    values = series.values.reshape(-1, 1)
    
    if method == 'minmax':
        # Min-max scaling to [0, 1]
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(values).flatten()
        
    elif method == 'zscore':
        # Z-score standardization
        scaler = StandardScaler()
        normalized = scaler.fit_transform(values).flatten()
        
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return pd.Series(normalized, index=series.index)

def normalize_dataframe(df: pd.DataFrame, 
                      columns: List[str], 
                      method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize specified columns in a dataframe.
    
    Args:
        df: Input dataframe
        columns: List of columns to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with normalized columns
    """
    df_copy = df.copy()
    
    for col in columns:
        if col in df_copy.columns and pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = normalize_column(df_copy[col], method)
    
    return df_copy

def create_feature(df: pd.DataFrame, 
                 feature_name: str, 
                 expression: str) -> pd.DataFrame:
    """
    Create a new feature using an expression.
    
    Args:
        df: Input dataframe
        feature_name: Name of the new feature
        expression: Python expression to evaluate (references df columns)
        
    Returns:
        pd.DataFrame: Dataframe with new feature added
    """
    df_copy = df.copy()
    
    try:
        # Evaluate the expression (be careful with security!)
        # This is a simplified approach - in production, use a safer evaluation method
        df_copy[feature_name] = eval(expression, {"__builtins__": {}}, {"df": df, "np": np})
        return df_copy
    except Exception as e:
        print(f"Error creating feature: {str(e)}")
        return df

def select_features(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """
    Select specified columns from dataframe.
    
    Args:
        df: Input dataframe
        selected_columns: List of columns to keep
        
    Returns:
        pd.DataFrame: Dataframe with only selected columns
    """
    return df[selected_columns].copy()

def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit UI for data cleaning and transformation.
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataframe
    """
    st.title("Data Cleaning and Transformation")
    
    if df is None:
        st.warning("Please upload a dataset first.")
        return None
    
    # Create a copy to work with
    cleaned_df = df.copy()
    
    # Step 1: Detect and handle non-standard missing values
    non_standard_missing = detect_non_standard_missing(cleaned_df)
    
    if non_standard_missing:
        st.subheader("Non-Standard Missing Values Detected")
        st.write("The following values might represent missing data:")
        
        for col, values in non_standard_missing.items():
            st.write(f"**{col}**: {', '.join(str(v) for v in values)}")
        
        replace = st.checkbox("Replace these values with NaN", value=True)
        
        if replace:
            cleaned_df = replace_non_standard_missing(cleaned_df, non_standard_missing)
            st.success("Non-standard missing values have been replaced with NaN.")
    
    # Step 2: Handle missing values
    st.subheader("Handle Missing Values")
    
    # Display missing value counts
    missing_counts = cleaned_df.isna().sum()
    if missing_counts.sum() > 0:
        st.write("Missing value counts by column:")
        st.write(missing_counts[missing_counts > 0])
        
        method = st.radio(
            "How would you like to handle missing values?",
            ["Leave as NaN", "Replace with specific value", "Use nearest neighbor imputation"]
        )
        
        if method == "Replace with specific value":
            fill_value = st.text_input("Value to use for replacement:")
            try:
                # Try to convert to numeric if possible
                fill_value = float(fill_value) if fill_value.replace('.', '', 1).isdigit() else fill_value
            except:
                pass
            
            cleaned_df = handle_missing_values(cleaned_df, 'fill', fill_value)
            st.success(f"Missing values replaced with '{fill_value}'.")
            
        elif method == "Use nearest neighbor imputation":
            n_neighbors = st.slider("Number of neighbors to consider", 1, 10, 5)
            cleaned_df = handle_missing_values(cleaned_df, 'knn', n_neighbors=n_neighbors)
            st.success("Missing values imputed using KNN method.")
    else:
        st.write("No missing values detected in the dataset.")
    
    # Step 3: Convert data types
    st.subheader("Data Type Conversion")
    
    convert_types = st.checkbox("Auto-detect and convert data types", value=True)
    if convert_types:
        original_types = cleaned_df.dtypes
        cleaned_df = convert_datatypes(cleaned_df)
        
        # Show changes in data types
        changed_cols = original_types[original_types != cleaned_df.dtypes].index.tolist()
        if changed_cols:
            st.success(f"Data types automatically converted for {len(changed_cols)} columns.")
            conversion_df = pd.DataFrame({
                'Original Type': original_types[changed_cols],
                'New Type': cleaned_df.dtypes[changed_cols]
            })
            st.write(conversion_df)
    
    # Step 4: Normalize numeric columns
    st.subheader("Normalize Numeric Columns")
    
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        normalize = st.checkbox("Normalize numeric columns", value=False)
        
        if normalize:
            cols_to_normalize = st.multiselect(
                "Select columns to normalize",
                options=numeric_cols,
                default=numeric_cols
            )
            
            normalization_method = st.radio(
                "Select normalization method:",
                ["Min-Max Scaling (0-1)", "Z-Score Standardization"]
            )
            
            method = 'minmax' if "Min-Max" in normalization_method else 'zscore'
            
            if cols_to_normalize:
                cleaned_df = normalize_dataframe(cleaned_df, cols_to_normalize, method)
                st.success(f"Normalized {len(cols_to_normalize)} columns using {normalization_method}.")
    
    # Step 5: Create new features
    st.subheader("Create New Features")
    
    create_new = st.checkbox("Create new features from existing ones", value=False)
    
    if create_new:
        col1, col2 = st.columns(2)
        
        with col1:
            new_feature_name = st.text_input("New feature name:")
        
        with col2:
            feature_expr = st.text_input("Expression (e.g., df['height'] / df['weight']):")
        
        if st.button("Add Feature") and new_feature_name and feature_expr:
            try:
                temp_df = create_feature(cleaned_df, new_feature_name, feature_expr)
                
                # Check if the feature was successfully added
                if new_feature_name in temp_df.columns and new_feature_name not in cleaned_df.columns:
                    cleaned_df = temp_df
                    st.success(f"Feature '{new_feature_name}' successfully created.")
                else:
                    st.error("Failed to create new feature.")
            except Exception as e:
                st.error(f"Error creating feature: {str(e)}")
    
    # Step 6: Select features
    st.subheader("Select Features")
    
    all_columns = cleaned_df.columns.tolist()
    features_to_keep = st.multiselect(
        "Select features to include in the analysis:",
        options=all_columns,
        default=all_columns
    )
    
    if features_to_keep:
        cleaned_df = select_features(cleaned_df, features_to_keep)
    
    # Show the resulting dataframe
    st.subheader("Cleaned Data Preview")
    st.dataframe(cleaned_df.head())
    
    # Option to download the cleaned dataset
    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=cleaned_df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv',
    )
    
    return cleaned_df
