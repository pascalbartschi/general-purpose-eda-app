"""
Module for handling data upload functionality of the EDA app.

This module provides comprehensive data upload and dataset management functionality including:
1. File upload with validation (type, size, content)
2. Sample dataset selection from a configured directory
3. Dataframe initialization and storage in session state
4. Session state management for coordination with other app modules

The upload module serves as the entry point for all data in the application
and ensures proper initialization of session state variables that will be used
by other modules (cleaning, EDA, reporting, etc.).

All upload operations include robust session state cleanup to ensure that uploading
a new dataset completely resets the application state, preventing any stale data 
from persisting between uploads.

Author: General Purpose EDA Team
"""
from typing import Tuple, Dict, Optional, Any
import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path
import io
import os

def get_available_datasets(data_dir: str = 'data') -> list:
    """
    Get a list of available datasets in the data directory.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        list: List of available dataset files
    """
    try:
        # Get the absolute path to ensure consistency
        abs_data_dir = Path(data_dir).resolve()
        
        if not abs_data_dir.exists() or not abs_data_dir.is_dir():
            return []
            
        # Get all files with supported extensions
        supported_extensions = ['.csv', '.xlsx', '.xls']
        datasets = []
        
        for file in abs_data_dir.iterdir():
            if file.is_file() and file.suffix.lower() in supported_extensions:
                datasets.append(file.name)
                
        return sorted(datasets)
    except Exception as e:
        st.error(f"Error reading data directory: {str(e)}")
        return []

def validate_file_type(file: Any) -> bool:
    """
    Validate if the uploaded file has an acceptable extension.
    
    Args:
        file: Streamlit file uploader object
        
    Returns:
        bool: True if file is of acceptable type, False otherwise
    """
    acceptable_extensions = ['.csv', '.xlsx', '.xls']
    file_extension = Path(file.name).suffix.lower()
    return file_extension in acceptable_extensions

def validate_file_size(file: Any, max_size_gb: float = 5.0) -> bool:
    """
    Validate if the uploaded file does not exceed max size.
    
    Args:
        file: Streamlit file uploader object
        max_size_gb: Maximum allowed file size in gigabytes
        
    Returns:
        bool: True if file size is acceptable, False otherwise
    """
    bytes_content = file.getvalue()
    file_size_gb = len(bytes_content) / (1024**3)  # Convert to GB
    return file_size_gb <= max_size_gb

def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate if the dataframe has at least one numeric column.
    
    Args:
        df: Pandas dataframe to validate
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    if df.empty:
        return False, "The uploaded file contains no data."
    
    numeric_columns = df.select_dtypes(include=['number']).columns
    if len(numeric_columns) == 0:
        return False, "The dataset must contain at least one numeric column."
    
    return True, "Dataset is valid."

def load_file(file: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a file into a pandas dataframe.
    
    Args:
        file: Streamlit file uploader object
        
    Returns:
        Tuple[pd.DataFrame, str]: (dataframe, error_message)
    """
    try:
        file_extension = Path(file.name).suffix.lower()
        
        if file_extension == '.csv':
            # Try to infer CSV delimiter
            df = pd.read_csv(file, sep=None, engine='python')
        elif file_extension == '.xlsx':
            df = pd.read_excel(file, engine='openpyxl')
        elif file_extension == '.xls':
            df = pd.read_excel(file, engine='xlrd')
        else:
            return None, f"Unsupported file extension: {file_extension}"
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def load_dataset_from_directory(filename: str, data_dir: str = 'data') -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a dataset from the data directory.
    
    Args:
        filename: Name of the file to load
        data_dir: Path to the data directory
        
    Returns:
        Tuple[pd.DataFrame, str]: (dataframe, error_message)
    """
    try:
        filepath = Path(data_dir) / filename
        
        file_extension = filepath.suffix.lower()
        if file_extension == '.csv':
            df = pd.read_csv(filepath, sep=None, engine='python')
        elif file_extension == '.xlsx':
            df = pd.read_excel(filepath, engine='openpyxl')
        elif file_extension == '.xls':
            df = pd.read_excel(filepath, engine='xlrd')
        else:
            return None, f"Unsupported file extension: {file_extension}"
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"

def upload_data() -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Handle file upload, validation, and initial loading.
    
    This function provides a user interface for uploading data files or selecting sample datasets.
    It validates inputs, loads data into pandas DataFrames, and manages session state to ensure
    proper coordination with other app modules (cleaning, EDA, etc.).
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: A tuple containing:
            - The loaded dataframe (or None if upload failed)
            - A dictionary with session state information including:
                - upload_success: Boolean indicating if upload was successful
                - df: The loaded dataframe
                - error_message: Any error message that occurred during upload
                - filename: The name of the uploaded/selected file
    """
    # Initialize local session state with default values
    session_state = {
        'upload_success': False,  # Track if upload was successful
        'df': None,               # The actual dataframe
        'error_message': None,    # Store any error messages
        'filename': None          # The name of the uploaded file
    }
    
    st.title("Data Upload")
    
    # Function to completely clear session state when new data is uploaded
    def clear_all_dataset_related_state():
        """
        Clear all session state variables related to datasets across all modules.
        
        This is a critical function that ensures uploading a new dataset completely 
        resets the app state, preventing any data or metadata from previous sessions 
        from persisting and potentially causing inconsistencies or unexpected behavior.
        
        The function:
        1. Removes all explicitly listed dataset-related variables
        2. Cleans up any additional state variables (except navigation state)
        3. Ensures no stale data persists between dataset uploads
        
        This approach solves common issues in stateful Streamlit apps where previous
        session data could affect new data processing pipelines.
        """
        # List all potential state variables that might contain dataset information
        # This is comprehensive to ensure no data leakage between datasets
        dataset_state_vars = [
            # Core app variables - primary dataset representations
            'df', 'original_df',
            
            # Cleaning module variables - used for transformation tracking and reset functionality
            'true_original_df',  # Original untouched data (used for reset functionality)
            'cleaned_df',        # Working dataset with all transformations applied
            'features_to_keep',  # Selected features/columns list
            
            # EDA module variables - results of analyses
            'basic_eda_results',    # Basic stats, summaries, and plots
            'advanced_eda_results', # Complex analyses results
            
            # Any cached dataframe variables
            'cleaning_steps'        # Record of applied transformations
        ]
        
        # Remove each dataset-related variable from session state
        # This loop is explicit to ensure critical variables are always cleared
        for var in dataset_state_vars:
            if var in st.session_state:
                del st.session_state[var]
                
        # Also delete any other session state keys that might be related to datasets
        # (except for navigation state like 'page' to maintain app navigation)
        # This catch-all approach ensures complete cleanup of any dynamically created state
        keys_to_keep = ['page']  # Variables that should persist across dataset changes
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
        for key in keys_to_delete:
            del st.session_state[key]
    
    # Create tabs for different data sources - this provides a clean UI with two import options
    upload_tab, sample_tab = st.tabs(["Upload Your Data", "Use Sample Dataset"])
    
    # Tab for uploading new data - handles user's own files
    with upload_tab:
        uploaded_file = st.file_uploader("Choose a data file", type=['csv', 'xlsx', 'xls'])
    
    # Tab for selecting sample datasets - provides pre-loaded examples for users to explore
    with sample_tab:
        data_dir = 'data'  # Default data directory where sample datasets are stored
        available_datasets = get_available_datasets(data_dir)
        
        if available_datasets:
            st.write("### Available Sample Datasets")
            # Create options for the selectbox
            dataset_options = {filename: filename for filename in available_datasets}
            selected_dataset = st.selectbox("Choose a sample dataset", options=list(dataset_options.keys()))
            
            if st.button("Load Selected Dataset"):
                # Step 1: Clear all existing dataset-related state
                # This is critical to prevent any stale data from previous uploads/sessions
                # affecting the new dataset. This ensures a clean slate for each upload.
                clear_all_dataset_related_state()
                
                # Step 2: Load the selected sample dataset from the configured data directory
                # We use a dedicated function that handles various file formats and errors
                df, error = load_dataset_from_directory(selected_dataset, data_dir)
                
                # Check if there was an error during loading
                if error:
                    # Handle loading errors gracefully with user feedback
                    session_state['error_message'] = error
                    st.error(error)
                else:
                    # Step 3: Update the session state with the newly loaded data
                    # This comprehensive state setup ensures all app modules can access the data
                    
                    # Update the local session state dictionary for return value
                    session_state['upload_success'] = True
                    session_state['df'] = df
                    session_state['filename'] = selected_dataset
                    
                    # Initialize all required session state variables with COPIES of the dataframe:
                    # Using .copy() is crucial to prevent inadvertent modifications and side effects
                    
                    # 1. Main dataframe for the app - primary working dataset
                    st.session_state['df'] = df.copy()
                    
                    # 2. Original unmodified dataframe for reference
                    # This preserves the initial state for comparison
                    st.session_state['original_df'] = df.copy()
                    
                    # 3. True original dataframe (specifically used by cleaning module for reset function)
                    # This is the anchor point that "Reset All" will revert to
                    st.session_state['true_original_df'] = df.copy()
                    
                    # 4. Cleaned dataframe that will store all cleaning transformations
                    # This is the working dataset used by the cleaning module
                    st.session_state['cleaned_df'] = df.copy()
                    
                    # 5. List of features/columns to keep (for feature selection)
                    # This initializes feature selection to include all columns
                    st.session_state['features_to_keep'] = list(df.columns)
                    
                    # Step 4: Display feedback and dataset preview to the user
                    # This provides immediate confirmation and insight into the loaded data
                    st.success(f"Dataset '{selected_dataset}' successfully loaded! Previous dataset has been cleared.")
                    
                    # Show a preview of the first few rows
                    st.write("Data Preview:")
                    st.dataframe(df.head())
                    
                    # Show technical dataset information (types, memory usage, etc.)
                    # This helps users understand the structure of their data
                    st.write("Basic Information:")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    
                    # Return the loaded dataframe and session state dictionary
                    # This makes the loaded data available to the main application
                    return df, session_state
        else:
            st.write("No sample datasets found in the data directory.")
    
    # Handle user-uploaded file if one is provided
    if uploaded_file is not None:
        # Step 1: Clear all session state to ensure a completely fresh start
        # This prevents any data from previous uploads from affecting the current upload
        clear_all_dataset_related_state()
        
        # Step 2: Multi-stage validation pipeline before accepting the file
        
        # 2.1: Validate file type/extension (security and compatibility check)
        # Only accept approved file formats to prevent errors and security issues
        if not validate_file_type(uploaded_file):
            session_state['error_message'] = f"Invalid file type. Please upload a .csv, .xlsx, or .xls file."
            st.error(session_state['error_message'])
            return None, session_state
        
        # 2.2: Validate file size to prevent memory issues or app crashes
        # This protects both the server and ensures reasonable processing times
        if not validate_file_size(uploaded_file):
            session_state['error_message'] = f"File size exceeds the 5GB limit."
            st.error(session_state['error_message'])
            return None, session_state
        
        # Step 3: Attempt to load the data into a pandas DataFrame
        # Our load_file function handles different formats and provides detailed error messages
        df, error = load_file(uploaded_file)
        if error:
            # Handle loading errors (e.g., malformed CSV, corrupted Excel)
            session_state['error_message'] = error
            st.error(error)
            return None, session_state
        
        # Step 4: Validate the dataframe content
        # Check for minimum requirements (e.g., must have at least one numeric column for EDA)
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            session_state['error_message'] = message
            st.error(message)
            return None, session_state
        
        # Step 5: Update session state with successfully loaded data
        # This section performs comprehensive state initialization for the entire app
        
        # Update local session state dictionary for return value
        session_state['upload_success'] = True
        session_state['df'] = df
        session_state['filename'] = uploaded_file.name
        
        # Store in Streamlit's session state with COPIES of the dataframe
        # Using .copy() is critical to prevent inadvertent modifications across modules
        
        # Main app dataframe - primary working copy
        st.session_state['df'] = df.copy()
        
        # Original unmodified reference copy
        st.session_state['original_df'] = df.copy()
        
        # Initialize cleaning module state variables
        # true_original_df: Anchor point for the "Reset All" functionality
        st.session_state['true_original_df'] = df.copy()
        
        # cleaned_df: Working copy for the cleaning module
        st.session_state['cleaned_df'] = df.copy()
        
        # List of features to include in feature selection
        st.session_state['features_to_keep'] = list(df.columns)
        
        # Step 6: Display feedback and dataset information to the user
        
        # Success message with the filename
        st.success(f"File '{uploaded_file.name}' successfully uploaded! Previous dataset has been cleared.")
        
        # Show a preview of the first few rows
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Display technical dataset information (types, memory usage, etc.)
        st.write("Basic Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Return the loaded dataframe and session state dictionary for use by the main app
        return df, session_state
    
    return None, session_state
