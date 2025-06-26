"""
Main module for the Streamlit EDA app.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import pickle

# Import app modules
from core.upload import upload_data
from core.cleaning import clean_data_ui
from core.eda_basic import basic_eda_ui
from core.eda_advanced import advanced_eda_ui
from core.report import report_generation_ui

# Set page config
st.set_page_config(
    page_title="EDA App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'df' not in st.session_state:
        st.session_state['df'] = None
    
    if 'original_df' not in st.session_state:
        st.session_state['original_df'] = None
        
    if 'cleaning_steps' not in st.session_state:
        st.session_state['cleaning_steps'] = []
        
    if 'basic_eda_results' not in st.session_state:
        st.session_state['basic_eda_results'] = {'visualizations': {}}
        
    if 'advanced_eda_results' not in st.session_state:
        st.session_state['advanced_eda_results'] = {'visualizations': {}}
        
    if 'page' not in st.session_state:
        st.session_state['page'] = 'upload'

def save_current_state(auto_save=True):
    """Save current state to an auto-save file."""
    if not auto_save:
        return
        
    try:
        session_data = {
            'df': st.session_state.get('df'),
            'original_df': st.session_state.get('original_df'),
            'cleaning_steps': st.session_state.get('cleaning_steps'),
            'basic_eda_results': st.session_state.get('basic_eda_results'),
            'advanced_eda_results': st.session_state.get('advanced_eda_results')
        }
        
        # Create directory if it doesn't exist
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        
        # Save to file
        with open(cache_dir / "auto_save.pkl", 'wb') as f:
            pickle.dump(session_data, f)
    except Exception as e:
        print(f"Error saving state: {str(e)}")

def load_auto_saved_state():
    """Load auto-saved state if available."""
    try:
        cache_file = Path("./cache/auto_save.pkl")
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                session_data = pickle.load(f)
                
            # Update session state
            if session_data.get('df') is not None:
                st.session_state['df'] = session_data['df']
                
            if session_data.get('original_df') is not None:
                st.session_state['original_df'] = session_data['original_df']
                
            if session_data.get('cleaning_steps') is not None:
                st.session_state['cleaning_steps'] = session_data['cleaning_steps']
                
            if session_data.get('basic_eda_results') is not None:
                st.session_state['basic_eda_results'] = session_data['basic_eda_results']
                
            if session_data.get('advanced_eda_results') is not None:
                st.session_state['advanced_eda_results'] = session_data['advanced_eda_results']
                
            return True
    except Exception as e:
        print(f"Error loading auto-saved state: {str(e)}")
        
    return False

def main():
    """Main function to run the app."""
    # Initialize session state
    initialize_session_state()
    
    # Try to load auto-saved state
    loaded = load_auto_saved_state()
    
    # App title and header
    st.sidebar.title("EDA App")
    
    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Upload", "Data Cleaning", "Basic EDA", "Advanced EDA", "Report Generation"],
        index=["upload", "cleaning", "basic_eda", "advanced_eda", "report"].index(st.session_state['page']) 
        if st.session_state['page'] in ["upload", "cleaning", "basic_eda", "advanced_eda", "report"] 
        else 0
    )
    
    # Map selection to session state
    page_map = {
        "Data Upload": "upload",
        "Data Cleaning": "cleaning",
        "Basic EDA": "basic_eda",
        "Advanced EDA": "advanced_eda",
        "Report Generation": "report"
    }
    st.session_state['page'] = page_map[page]
    
    # Display different pages based on selection
    if page == "Data Upload":
        df, session_state = upload_data()
        
        if session_state['upload_success'] and df is not None:
            # Store the original dataframe and ensure it's also ready for cleaning module
            st.session_state['original_df'] = df.copy()
            st.session_state['df'] = df.copy()
            
            # Initialize cleaning-specific session state if not already done in upload_data
            if 'true_original_df' not in st.session_state:
                st.session_state['true_original_df'] = df.copy()
            if 'cleaned_df' not in st.session_state:
                st.session_state['cleaned_df'] = df.copy()
            if 'features_to_keep' not in st.session_state:
                st.session_state['features_to_keep'] = list(df.columns)
    
    elif page == "Data Cleaning":
        if st.session_state.get('df') is not None:
            # Set cleaning page flag
            st.session_state['page'] = 'cleaning'
            
            # Initialize true_original_df if it doesn't exist yet
            if 'true_original_df' not in st.session_state:
                st.session_state['true_original_df'] = st.session_state['df'].copy()
                
            # Initialize cleaned_df if it doesn't exist yet
            if 'cleaned_df' not in st.session_state:
                st.session_state['cleaned_df'] = st.session_state['df'].copy()
                
            # Initialize features_to_keep
            if 'features_to_keep' not in st.session_state:
                st.session_state['features_to_keep'] = list(st.session_state['df'].columns)
            
            # Pass the dataframe to the cleaning UI
            result_df = clean_data_ui(st.session_state['df'])
            
            # Update the main dataframe with the result from cleaning
            if result_df is not None:
                st.session_state['df'] = result_df
            
            # Record cleaning steps (this would be updated by the clean_data_ui function)
            # For simplicity, we're just adding a placeholder step here
            if 'cleaning_steps' not in st.session_state:
                st.session_state['cleaning_steps'] = []
                
            # Example cleaning step - in a real app this would be more detailed
            st.session_state['cleaning_steps'].append({
                'action': 'Data Cleaning',
                'description': 'Cleaned and transformed data'
            })
        else:
            st.warning("Please upload a dataset first.")
    
    elif page == "Basic EDA":
        if st.session_state['df'] is not None:
            basic_eda_ui(st.session_state['df'])
        else:
            st.warning("Please upload and clean a dataset first.")
    
    elif page == "Advanced EDA":
        if st.session_state['df'] is not None:
            advanced_eda_ui(st.session_state['df'])
        else:
            st.warning("Please upload and clean a dataset first.")
    
    elif page == "Report Generation":
        if st.session_state['df'] is not None:
            report_generation_ui(
                st.session_state['df'],
                st.session_state['cleaning_steps'],
                st.session_state['basic_eda_results'],
                st.session_state['advanced_eda_results']
            )
        else:
            st.warning("Please upload, clean, and analyze a dataset first.")
    
    # Auto-save state
    save_current_state()
    
    # Display sidebar info based on loaded state
    if loaded:
        if st.session_state['df'] is not None:
            st.sidebar.success("Session loaded from auto-save.")
            st.sidebar.write(f"Dataset: {st.session_state['df'].shape[0]} rows × {st.session_state['df'].shape[1]} columns")

if __name__ == "__main__":
    main()
