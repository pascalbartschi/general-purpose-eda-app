"""
Module for handling data upload functionality of the EDA app.
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
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: (dataframe, session_state)
    """
    session_state = {
        'upload_success': False,
        'df': None,
        'error_message': None,
        'filename': None    }
    
    st.title("Data Upload")
    
    # Create tabs for different data sources
    upload_tab, sample_tab = st.tabs(["Upload Your Data", "Use Sample Dataset"])
    
    # Tab for uploading new data
    with upload_tab:
        uploaded_file = st.file_uploader("Choose a data file", type=['csv', 'xlsx', 'xls'])
    
    # Tab for selecting sample datasets
    with sample_tab:
        data_dir = 'data'  # Default data directory
        available_datasets = get_available_datasets(data_dir)
        
        if available_datasets:
            st.write("### Available Sample Datasets")
            dataset_options = {filename: filename for filename in available_datasets}
            selected_dataset = st.selectbox("Choose a sample dataset", options=list(dataset_options.keys()))
            
            if st.button("Load Selected Dataset"):
                df, error = load_dataset_from_directory(selected_dataset, data_dir)
                if error:
                    session_state['error_message'] = error
                    st.error(error)
                else:
                    # Update session state with successful load
                    session_state['upload_success'] = True
                    session_state['df'] = df
                    session_state['filename'] = selected_dataset
                    
                    # Display success message and preview
                    st.success(f"Dataset '{selected_dataset}' successfully loaded!")
                    st.write("Data Preview:")
                    st.dataframe(df.head())
                    
                    # Display basic info
                    st.write("Basic Information:")
                    buffer = io.StringIO()
                    df.info(buf=buffer)
                    st.text(buffer.getvalue())
                    
                    return df, session_state
        else:
            st.write("No sample datasets found in the data directory.")
    
    if uploaded_file is not None:
        # Validate file type
        if not validate_file_type(uploaded_file):
            session_state['error_message'] = f"Invalid file type. Please upload a .csv, .xlsx, or .xls file."
            st.error(session_state['error_message'])
            return None, session_state
        
        # Validate file size
        if not validate_file_size(uploaded_file):
            session_state['error_message'] = f"File size exceeds the 5GB limit."
            st.error(session_state['error_message'])
            return None, session_state
        
        # Load data
        df, error = load_file(uploaded_file)
        if error:
            session_state['error_message'] = error
            st.error(error)
            return None, session_state
        
        # Validate dataframe
        is_valid, message = validate_dataframe(df)
        if not is_valid:
            session_state['error_message'] = message
            st.error(message)
            return None, session_state
        
        # Update session state with successful upload
        session_state['upload_success'] = True
        session_state['df'] = df
        session_state['filename'] = uploaded_file.name
        
        # Display success message and preview
        st.success(f"File '{uploaded_file.name}' successfully uploaded!")
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        # Display basic info
        st.write("Basic Information:")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        return df, session_state
    
    return None, session_state
