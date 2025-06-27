"""
Module for data cleaning and transformation functions of the EDA app.

This module provides comprehensive data cleaning and transformation capabilities including:
1. Detection and handling of non-standard missing values
2. Missing value imputation (categorical and numeric) using various strategies
3. Datatype conversion and optimization
4. Feature normalization and standardization
5. Feature creation and selection

The module handles the UI components for these operations, but also provides standalone
functions that can be imported for use in other modules or scripts.

All operations maintain clean session state management to enable proper reset functionality
and coordination with other app components.

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
    
    This function identifies potential non-standard missing values in string/object columns
    such as empty strings, dashes, periods, etc. that should be interpreted as NaN. This is 
    a common data quality issue in real-world datasets, especially those exported from systems 
    that represent nulls in different ways.
    
    The function only checks object/string columns for efficiency, and uses a curated list
    of common non-standard missing value representations.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict[str, List[Any]]: Dictionary with columns as keys and lists of detected non-standard nulls
    
    Example:
        >>> df = pd.DataFrame({'col1': ['a', '', 'c'], 'col2': [1, 2, '-']})
        >>> detect_non_standard_missing(df)
        {'col1': [''], 'col2': ['-']}
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
    
    This function standardizes the representation of missing values by converting all 
    non-standard missing values (as identified by detect_non_standard_missing) to pandas' 
    standard NaN. This is a critical data cleaning step that enables proper handling by 
    pandas functions and statistical analyses.
    
    The function creates a copy of the dataframe to avoid modifying the original, following
    the principle of non-destructive data transformations.
    
    Args:
        df: Input dataframe
        null_values_dict: Dictionary with columns as keys and lists of values to replace
        
    Returns:
        pd.DataFrame: Dataframe with non-standard nulls replaced with NaN
        
    Example:
        >>> df = pd.DataFrame({'col1': ['a', '', 'c'], 'col2': [1, 2, '-']})
        >>> nulls = {'col1': [''], 'col2': ['-']}
        >>> replace_non_standard_missing(df, nulls)
           col1  col2
        0    a   1.0
        1  NaN   2.0
        2    c   NaN
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
    Handle missing values in the dataframe using various imputation strategies.
    
    This function provides multiple strategies for handling missing values:
    - 'leave': Keep missing values as NaN (useful when downstream methods handle NaNs)
    - 'fill': Replace NaNs with a specific value (mean, median, mode, or custom value)
    - 'knn': Use K-Nearest Neighbors algorithm to impute based on similar observations
    
    KNN imputation is a sophisticated approach that uses the values of nearest neighbors
    in feature space to estimate missing values. It works particularly well when there 
    are correlations between features and when missing data is not completely random.
    
    Args:
        df: Input dataframe
        method: Method to handle missing values ('leave', 'fill', 'knn')
        fill_value: Value to use if method is 'fill'
        n_neighbors: Number of neighbors for KNN imputation (higher=smoother, lower=more local)
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
        
    Raises:
        ValueError: If an unsupported imputation method is provided
        
    Notes:
        - KNN imputation is only applied to numeric columns
        - Non-numeric columns need to be handled separately (typically through mode imputation)
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
    Convert and infer appropriate data types for columns to optimize memory usage and performance.
    
    This function uses smart heuristics to:
    1. Convert string columns to numeric types where appropriate
    2. Convert string columns to datetime types where appropriate
    3. Convert low-cardinality string columns to categorical type for memory efficiency
    
    Memory optimization is particularly important for large datasets, as categorical and
    optimized numeric types can significantly reduce memory usage. Additionally, proper
    type conversion ensures that analytical operations use appropriate methods based on
    data semantics (e.g., datetime operations for date columns).
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with inferred and optimized data types
        
    Notes:
        - Conversion to numeric/datetime is only applied if <30% of values become NaN
        - Categorical conversion is applied to columns with cardinality <5% of row count
        - The function creates a copy of the input dataframe to avoid side effects
        - This conversion can significantly improve memory usage and performance
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
    Normalize a numerical series using specified scaling method.
    
    Normalization is crucial for:
    1. Machine learning algorithms that are sensitive to feature scales
    2. Comparing features with different units/scales
    3. Gradient-based optimization that converges faster with normalized data
    
    Two methods are supported:
    - 'minmax': Scales values to range [0,1], preserves distribution shape but compresses outliers
    - 'zscore': Standardizes to mean=0, std=1, preserves outlier influence but not original units
    
    Args:
        series: Input series with numerical values
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        pd.Series: Normalized series with same index as input
        
    Raises:
        ValueError: If an unsupported normalization method is specified
        
    Notes:
        - Non-numeric series are returned unchanged with a warning
        - NaN values remain NaN after normalization
        - Uses scikit-learn's robust scaling implementations
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
    Normalize specified columns in a dataframe while preserving non-selected columns.
    
    This is a higher-level function that applies column-wise normalization to multiple
    specified columns in a dataframe. It's particularly useful in preparing data for:
    - Machine learning models that require normalized features
    - Visualizations where comparing features with different scales is important
    - Statistical analyses where standardized units are preferred
    
    The function only affects specified numeric columns and leaves other columns untouched,
    making it safe to use on mixed-type dataframes.
    
    Args:
        df: Input dataframe
        columns: List of columns to normalize
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        pd.DataFrame: Dataframe with normalized columns
        
    Notes:
        - Creates a copy of the input dataframe to avoid side effects
        - Silently skips any specified columns that aren't numeric
        - Each column is normalized independently, so relative scales between 
          columns may not be preserved
        - Consider using PCA for maintaining relative importance of features
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
    Create a new feature using a Python expression evaluated against the dataframe.
    
    Feature engineering is a critical part of the data science workflow, often making
    the difference between average and exceptional model performance. This function
    allows data scientists to create new features using Python expressions that can 
    reference existing columns and NumPy functions.
    
    Common use cases include:
    - Creating interaction terms (e.g., 'df.height * df.weight')
    - Computing derived metrics (e.g., 'df.revenue / df.cost')
    - Applying transformations (e.g., 'np.log(df.price)')
    - Creating polynomial features (e.g., 'df.x**2')
    
    Args:
        df: Input dataframe
        feature_name: Name of the new feature column
        expression: Python expression to evaluate (can reference df columns and np functions)
        
    Returns:
        pd.DataFrame: Dataframe with new feature added
        
    Warning:
        Using eval() with user input is potentially unsafe. In production environments,
        consider using a more restricted expression parser or validation system.
        
    Notes:
        - The expression has access to the dataframe as 'df' and NumPy as 'np'
        - Other built-ins are restricted for security
        - Errors in expression evaluation are caught and printed
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
    Select specified columns from dataframe to create a filtered dataset.
    
    Feature selection is an important step in the data preparation pipeline that:
    1. Reduces dimensionality to improve model performance and interpretability
    2. Removes irrelevant or redundant features
    3. Focuses the analysis on the most important variables
    4. Can reduce computational resources needed for downstream processing
    
    This function implements the most basic form of feature selection - manual selection
    based on domain knowledge or preliminary analysis. More advanced methods might use
    statistical tests, model-based importance, or dimensionality reduction techniques.
    
    Args:
        df: Input dataframe
        selected_columns: List of columns to keep in the resulting dataframe
        
    Returns:
        pd.DataFrame: Dataframe containing only the selected columns
        
    Notes:
        - Returns a copy of the input dataframe to avoid modifying the original
        - If selected_columns is empty, returns an empty dataframe
        - If columns in selected_columns don't exist in df, will raise KeyError
    """
    return df[selected_columns].copy()

def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit UI for data cleaning and transformation with comprehensive session state management.
    
    This function provides a complete interactive interface for data cleaning operations, 
    organized in a logical, step-by-step workflow:
    
    1. Non-standard missing value detection and handling
    2. Missing value imputation (separate workflows for categorical and numeric data)
    3. Data type conversion and optimization
    4. Feature normalization/standardization
    5. Feature selection
    
    Each operation maintains proper session state to ensure:
    - Changes are immediately reflected in the UI
    - The "Reset All" functionality works correctly
    - State is preserved between page navigations
    - Data transformations are applied in the correct order
    
    Args:
        df: Input dataframe (typically from upload module)
        
    Returns:
        pd.DataFrame: Cleaned and transformed dataframe
        
    Notes:
        - Session state variables used:
          * true_original_df: Original data, preserved for reset functionality
          * cleaned_df: Current working dataset with all transformations applied
          * features_to_keep: Current selected features for feature selection step
          * page: Current app page/tab
        - UI is refreshed after each operation using st.experimental_rerun()
        - All operations work on copies of dataframes to prevent side effects
    """
    st.title("Data Cleaning and Transformation")
    
    if df is None:
        st.warning("Please upload a dataset first.")
        return None

    # --- Session State Initialization ---
    # First, ensure we're set to stay on the cleaning page
    st.session_state['page'] = 'cleaning'
    
    # Make sure we have all required session state variables
    if 'true_original_df' not in st.session_state:
        st.session_state['true_original_df'] = df.copy()
        
    if 'cleaned_df' not in st.session_state:
        st.session_state['cleaned_df'] = df.copy()
        
    if 'features_to_keep' not in st.session_state:
        st.session_state['features_to_keep'] = list(df.columns)

    # --- Reset Button ---
    # This button allows users to reset all cleaning operations and start fresh
    # without losing the original dataset or having to go back to the upload page
    reset_clicked = st.button("Reset All Cleaning Steps")
    if reset_clicked:
        # The reset mechanism is intentionally simple and direct:
        # 1. Restore from true_original_df (which is preserved from initial upload)
        # 2. Update all cleaning-related session state
        # 3. Force UI refresh to show reset data
        if 'true_original_df' in st.session_state:
            # Reset the working dataset to original uploaded state
            st.session_state.cleaned_df = st.session_state.true_original_df.copy()
            # Reset feature selection to include all original columns
            st.session_state.features_to_keep = list(st.session_state.true_original_df.columns)
            
            # Provide feedback to the user
            st.success("All cleaning steps have been reset. Original dataset restored.")
            
            # Explicitly set the page to ensure we stay on the cleaning page after reset
            # This is crucial to avoid unintended navigation back to the upload screen
            st.session_state.page = 'cleaning'
            
            # Force a complete UI refresh to reflect the reset state
            # This ensures all widgets display the correct initial values
            st.experimental_rerun()
        else:
            # This error occurs if the session state is corrupted or if
            # the app was accessed directly without going through the upload process
            st.error("Could not find the original dataset. Please try uploading your data again.")

    # Use the dataframe from session state for all operations
    cleaned_df = st.session_state.cleaned_df

    # Step 1: Non-Standard Missing Values
    # This step identifies and standardizes various representations of missing values 
    # that might not be automatically recognized by pandas (e.g., "", "-", etc.)
    st.subheader("1. Non-Standard Missing Values")
    
    # Scan the dataframe for potential non-standard nulls using our detection function
    non_standard_missing = detect_non_standard_missing(cleaned_df)
    
    if non_standard_missing:
        # Found non-standard missing values - display them to the user for verification
        st.write("The following values might represent missing data:")
        for col, values in non_standard_missing.items():
            st.write(f"**{col}**: {', '.join(str(v) for v in values)}")
        
        # Provide a button to standardize these values to NaN
        # This is important for consistent handling in later steps (e.g., imputation)
        if st.button("Replace these values with NaN"):
            # Apply the replacement and update session state
            st.session_state.cleaned_df = replace_non_standard_missing(cleaned_df, non_standard_missing)
            st.success("Non-standard missing values have been replaced with NaN.")
            
            # Force UI refresh to show updated data and statistics
            st.experimental_rerun()
    else:
        # No issues found - this is good news for data quality
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
    # Feature selection is often one of the final steps in cleaning/preparation,
    # allowing analysts to focus on relevant variables and reduce dimensionality
    st.subheader("5. Feature Creation & Selection")
    
    # Feature Selection UI
    # We use a multiselect widget to let users choose which columns to keep
    # The default selection is based on either previous selections or all columns
    all_columns = cleaned_df.columns.tolist()
    default_selection = st.session_state.get('features_to_keep', all_columns)
    
    # Present multiselect with current columns
    features_to_keep = st.multiselect(
        "Select features to include in the final dataset:",
        options=all_columns,
        default=default_selection,
        help="Select columns to keep in your dataset. This can reduce dimensionality and focus analysis."
    )
    
    # Apply feature selection when button is clicked
    if st.button("Apply Feature Selection"):
        # Store selected features in session state for persistence between app reruns
        st.session_state.features_to_keep = features_to_keep
        
        # Update the dataframe to only include selected columns
        st.session_state.cleaned_df = select_features(cleaned_df, features_to_keep)
        st.success("Feature selection applied.")
        
        # Force UI refresh to show the updated dataset with selected features only
        st.experimental_rerun()

    # Make sure we're using the current cleaned_df
    cleaned_df = st.session_state.cleaned_df
    
    # Final Preview Section - Always show the current state of the cleaned dataframe
    # This section always reflects the latest state after all applied transformations
    st.subheader("Cleaned Data Preview")
    
    # Display the first few rows of the cleaned dataframe
    # This gives users immediate feedback on the effects of their cleaning operations
    st.dataframe(cleaned_df.head(), use_container_width=True)
    
    # Add information about dataset dimensions after cleaning
    st.info(f"Current dataset dimensions: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    
    # Download Button - Export the current cleaned dataset
    # This allows users to save their work at any point in the cleaning process
    st.download_button(
        label="Download Cleaned Dataset as CSV",
        data=cleaned_df.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv',
        key="download_csv_button",
        help="Download the current state of your cleaned dataset as a CSV file"
    )
    
    # Always return the current cleaned dataframe for use by other modules
    # This enables proper coordination between app components
    return cleaned_df
