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
    # A more focused list of potential nulls based on common data issues.
    # We are excluding values that pandas typically recognizes as NA by default (e.g., 'NA', 'N/A', 'null').
    # We are explicitly including '' to catch blank or whitespace-only strings.
    potential_nulls = ['', '-', '.']
    
    non_standard_nulls = {}
    
    # Iterate only over object columns for efficiency
    for col in df.select_dtypes(include=['object']).columns:
        # Get unique string values, strip whitespace, and convert to lower case for comparison
        unique_vals = df[col].dropna().unique()
        
        found_nulls = [val for val in unique_vals 
                       if isinstance(val, str) and val.strip().lower() in potential_nulls]
        
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
        # This part is now handled by the UI for more control
        # for col in df_copy.select_dtypes(exclude=['number']).columns:
        #     if df_copy[col].isna().any():
        #         mode_val = df_copy[col].mode()[0]
        #         df_copy[col] = df_copy[col].fillna(mode_val)
                
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
        # Try converting to numeric only for object type columns
        if df_copy[col].dtype == 'object':
            try:
                # Try converting to numeric
                numeric_version = pd.to_numeric(df_copy[col], errors='coerce')
                
                # Heuristic: if conversion to numeric does not make more than 30% of data NaN, we apply it.
                # This prevents converting purely categorical columns.
                if numeric_version.isna().mean() < 0.3:
                    df_copy[col] = numeric_version
                else:
                    # If numeric conversion fails, try datetime
                    datetime_version = pd.to_datetime(df_copy[col], errors='coerce')
                    # Apply same heuristic for datetime
                    if datetime_version.isna().mean() < 0.3:
                        df_copy[col] = datetime_version
            except (ValueError, TypeError):
                # If conversion fails, just continue
                pass
                
    # Convert remaining object columns with low cardinality to 'category'
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

    # --- Session State Initialization ---
    # Only set true_original_df if it does not exist (i.e., on first upload)
    if 'true_original_df' not in st.session_state:
        st.session_state.true_original_df = df.copy()
        print('DEBUG: true_original_df set.')
    # Only set original_df/cleaned_df if new data is uploaded
    if 'cleaned_df' not in st.session_state or not st.session_state.get('original_df', pd.DataFrame()).equals(df):
        st.session_state.original_df = df.copy()
        st.session_state.cleaned_df = df.copy()
        st.success("New dataset loaded. Cleaning state has been reset.")
        print('DEBUG: original_df and cleaned_df set.')

    # --- Reset Button ---
    if st.button("Reset All Cleaning Steps"):
        # Deep copy by serializing/deserializing to ensure breaking all references
        true_df_csv = st.session_state.true_original_df.to_csv(index=False)
        restored_df = pd.read_csv(pd.io.common.StringIO(true_df_csv))
        
        # Forcefully replace the dataframes
        st.session_state.original_df = restored_df
        st.session_state.cleaned_df = restored_df
        
        print(f"DEBUG: Reset - Original columns before: {list(st.session_state.true_original_df.columns)}")
        print(f"DEBUG: Reset - Cleaned after reset: {list(st.session_state.cleaned_df.columns)}")
        
        # Reset all session state variables
        keys_to_delete = []
        for key in st.session_state:
            if key not in ['true_original_df', 'original_df', 'cleaned_df']:
                keys_to_delete.append(key)
                
        for key in keys_to_delete:
            del st.session_state[key]
            
        st.session_state['features_to_keep'] = list(restored_df.columns)
        
        st.success("All cleaning steps have been reset. App will reload.")
        st.experimental_rerun()

    # Use the dataframe from session state for all operations
    cleaned_df = st.session_state.cleaned_df

    # Step 1: Non-Standard Missing Values
    st.subheader("1. Non-Standard Missing Values")
    non_standard_missing = detect_non_standard_missing(cleaned_df)
    if non_standard_missing:
        st.write("The following values might represent missing data:")
        for col, values in non_standard_missing.items():
            st.write(f"**{col}**: {', '.join(str(v) for v in values)}")
        
        if st.button("Replace these values with NaN"):
            st.session_state.cleaned_df = replace_non_standard_missing(cleaned_df, non_standard_missing)
            st.success("Non-standard missing values have been replaced with NaN.")
            st.experimental_rerun()
    else:
        st.write("No non-standard missing values detected.")

    # Step 2: Handle Missing Values
    st.subheader("2. Handle Missing Values")
    # Identify all columns with any form of missing data (NaN or non-standard)
    cols_with_nan = cleaned_df.columns[cleaned_df.isna().any()].tolist()
    cols_with_non_standard = list(non_standard_missing.keys())
    all_missing_cols = sorted(list(set(cols_with_nan + cols_with_non_standard)))

    numeric_missing_cols = [col for col in all_missing_cols if pd.api.types.is_numeric_dtype(cleaned_df[col])]
    non_numeric_missing_cols = [col for col in all_missing_cols if not pd.api.types.is_numeric_dtype(cleaned_df[col])]

    # --- UI for Categorical/Text Columns ---
    if non_numeric_missing_cols:
        st.markdown("#### In Categorical/Text Columns")
        st.write(f"The following columns have missing values: `{', '.join(non_numeric_missing_cols)}`")
        
        method_non_numeric = st.radio(
            "How would you like to handle these missing values?",
            ("Leave as NaN", "Replace with a specific value"),
            key="non_numeric_imputation_method"
        )

        if method_non_numeric == "Leave as NaN":
            st.info("This option ensures that all non-standard missing values (e.g., blanks, '-', '.') in the selected columns are converted to a standard NaN format.")
            cols_to_convert = st.multiselect(
                "Select columns to apply this conversion to:",
                options=non_numeric_missing_cols,
                default=non_numeric_missing_cols,
                key="select_non_numeric_convert_cols"
            )
            if st.button("Convert Non-Standard to NaN"):
                if cols_to_convert:
                    # Detect non-standard missing values only in the selected columns
                    df_subset = st.session_state.cleaned_df[cols_to_convert]
                    non_standard_to_convert = detect_non_standard_missing(df_subset)
                    
                    if non_standard_to_convert:
                        st.session_state.cleaned_df = replace_non_standard_missing(st.session_state.cleaned_df, non_standard_to_convert)
                        st.success(f"Non-standard missing values in selected columns converted to NaN.")
                        st.experimental_rerun()
                    else:
                        st.info("No non-standard missing values were found in the selected columns.")
                else:
                    st.warning("Please select at least one column.")

        elif method_non_numeric == "Replace with a specific value":
            cols_to_fill = st.multiselect(
                "Select columns to fill:",
                options=non_numeric_missing_cols,
                default=non_numeric_missing_cols,
                key="select_non_numeric_cols"
            )
            fill_value = st.text_input(
                "Value to replace NaNs with:",
                value="Unknown",
                key="fill_non_numeric_value"
            )
            if st.button("Apply to Categorical Columns"):
                if cols_to_fill:
                    temp_df = st.session_state.cleaned_df.copy()
                    for col in cols_to_fill:
                        if pd.api.types.is_categorical_dtype(temp_df[col]):
                            if fill_value not in temp_df[col].cat.categories:
                                temp_df[col] = temp_df[col].cat.add_categories([fill_value])
                        temp_df[col] = temp_df[col].fillna(fill_value)
                    st.session_state.cleaned_df = temp_df
                    st.success(f"Missing values in selected columns replaced with '{fill_value}'.")
                    st.experimental_rerun()

    # --- UI for Numeric Columns ---
    if numeric_missing_cols:
        st.markdown("#### In Numeric Columns")
        st.write(f"The following columns have missing values: `{', '.join(numeric_missing_cols)}`")

        method_numeric = st.radio(
            "How would you like to handle missing values in these numeric columns?",
            ("Leave as NaN", "Replace with a specific value", "Use nearest neighbor imputation (KNN)"),
            key="numeric_imputation_method"
        )

        if method_numeric == "Replace with a specific value":
            fill_value_numeric = st.text_input("Value to use for replacement (must be a number):", key="fill_numeric")
            if st.button("Apply to Numeric Columns"):
                try:
                    fill_val = float(fill_value_numeric)
                    st.session_state.cleaned_df[numeric_missing_cols] = st.session_state.cleaned_df[numeric_missing_cols].fillna(fill_val)
                    st.success(f"Missing values in numeric columns replaced with '{fill_val}'.")
                    st.experimental_rerun()
                except (ValueError, TypeError):
                    st.error("Please enter a valid number.")

        elif method_numeric == "Use nearest neighbor imputation (KNN)":
            n_neighbors = st.slider("Number of neighbors for KNN", 1, 15, 5, key="knn_neighbors")
            if st.button("Apply KNN Imputation"):
                st.session_state.cleaned_df = handle_missing_values(st.session_state.cleaned_df, 'knn', n_neighbors=n_neighbors)
                st.success("Missing values in numeric columns imputed using KNN.")
                st.experimental_rerun()

    if not all_missing_cols:
        st.write("No missing values detected in the dataset.")

    # Step 3: Data Type Conversion
    st.subheader("3. Data Type Conversion")
    if st.button("Auto-detect and Convert Data Types"):
        original_types = cleaned_df.dtypes.copy()
        st.session_state.cleaned_df = convert_datatypes(cleaned_df)
        new_types = st.session_state.cleaned_df.dtypes
        changed_cols = original_types[original_types != new_types].index.tolist()
        if changed_cols:
            st.success(f"Data types automatically converted for {len(changed_cols)} columns.")
        else:
            st.info("No data type conversions were necessary.")
        st.experimental_rerun()

    # Step 4: Normalize Numeric Columns
    st.subheader("4. Normalize Numeric Columns")
    numeric_cols = cleaned_df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        cols_to_normalize = st.multiselect(
            "Select columns to normalize",
            options=numeric_cols,
            key="normalize_multiselect"
        )
        if cols_to_normalize:
            normalization_method = st.radio(
                "Select normalization method:",
                ("Min-Max Scaling (0-1)", "Z-Score Standardization"),
                key="normalize_radio"
            )
            method = 'minmax' if "Min-Max" in normalization_method else 'zscore'
            if st.button("Apply Normalization"):
                st.session_state.cleaned_df = normalize_dataframe(cleaned_df, cols_to_normalize, method)
                st.success(f"Normalized {len(cols_to_normalize)} columns using {normalization_method}.")
                st.experimental_rerun()

    # Step 5: Feature Creation & Selection
    st.subheader("5. Feature Creation & Selection")
    
    # Feature Selection
    all_columns = cleaned_df.columns.tolist()
    default_selection = st.session_state.get('features_to_keep', all_columns)
    features_to_keep = st.multiselect(
        "Select features to include in the final dataset:",
        options=all_columns,
        default=default_selection
    )
    if st.button("Apply Feature Selection"):
        st.session_state.features_to_keep = features_to_keep
        st.session_state.cleaned_df = select_features(cleaned_df, features_to_keep)
        st.success("Feature selection applied.")
        st.experimental_rerun()

    # Final Preview
    st.subheader("Cleaned Data Preview")
    st.dataframe(st.session_state.cleaned_df.head())
    
    # Download Button
    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=st.session_state.cleaned_df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv',
        key="download_csv_button"
    )
    
    return st.session_state.cleaned_df
