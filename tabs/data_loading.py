import streamlit as st
import pandas as pd
from pathlib import Path
import io
import base64

def app():
    st.header("Data Loading and Filtering")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Read the data
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['data'] = df
            st.success(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Display data info
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Data filtering options
            st.subheader("Data Filtering")
            
            # Column selection
            all_columns = df.columns.tolist()
            selected_columns = st.multiselect("Select columns to keep", all_columns, default=all_columns)
            
            if len(selected_columns) > 0:
                filtered_df = df[selected_columns]
                
                # Filter by values for categorical columns
                categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
                if categorical_cols:
                    st.subheader("Filter by categorical values")
                    col_to_filter = st.selectbox("Select a column to filter", categorical_cols)
                    unique_values = filtered_df[col_to_filter].unique().tolist()
                    selected_values = st.multiselect("Select values to include", unique_values, default=unique_values)
                    
                    if selected_values:
                        filtered_df = filtered_df[filtered_df[col_to_filter].isin(selected_values)]
                
                # Numeric range filters
                numeric_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_cols:
                    st.subheader("Filter by numeric ranges")
                    col_to_filter_num = st.selectbox("Select a numeric column to filter", numeric_cols)
                    
                    min_val = float(filtered_df[col_to_filter_num].min())
                    max_val = float(filtered_df[col_to_filter_num].max())
                    
                    range_values = st.slider(
                        f"Select range for {col_to_filter_num}",
                        min_val, max_val, (min_val, max_val)
                    )
                    
                    filtered_df = filtered_df[
                        (filtered_df[col_to_filter_num] >= range_values[0]) & 
                        (filtered_df[col_to_filter_num] <= range_values[1])
                    ]
                
                # Save filtered data to session state
                st.session_state['filtered_data'] = filtered_df
                
                st.subheader("Filtered Data Preview")
                st.dataframe(filtered_df.head())
                
                # Download button for filtered data
                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                b64 = base64.b64encode(csv_str.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="filtered_data.csv">Download Filtered CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    else:
        st.info("Please upload a CSV file to get started")
