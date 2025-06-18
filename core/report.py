"""
Module for report generation functionality of the EDA app.
"""
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
import streamlit as st
import base64
import io
import json
import pickle
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import os
import jinja2
# Lazy import for weasyprint to avoid GTK dependency error
# import weasyprint
import openai
import time

def create_report_config(
    title: str = "EDA Report",
    author: str = "Streamlit EDA App",
    include_data_summary: bool = True,
    include_cleaning_summary: bool = True,
    include_basic_eda: bool = True,
    include_advanced_eda: bool = True,
    selected_visualizations: Optional[List[str]] = None,
    auto_generate_text: bool = True
) -> Dict[str, Any]:
    """
    Create a configuration for the report.
    
    Args:
        title: Report title
        author: Report author
        include_data_summary: Whether to include data summary section
        include_cleaning_summary: Whether to include data cleaning section
        include_basic_eda: Whether to include basic EDA section
        include_advanced_eda: Whether to include advanced EDA section
        selected_visualizations: List of selected visualization names to include
        auto_generate_text: Whether to auto-generate text explanations
        
    Returns:
        Dict[str, Any]: Report configuration dictionary
    """
    return {
        'title': title,
        'author': author,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'include_data_summary': include_data_summary,
        'include_cleaning_summary': include_cleaning_summary,
        'include_basic_eda': include_basic_eda,
        'include_advanced_eda': include_advanced_eda,
        'selected_visualizations': selected_visualizations or [],
        'auto_generate_text': auto_generate_text
    }

def fig_to_base64(fig) -> str:
    """
    Convert a matplotlib figure to base64 encoding for embedding in HTML.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        str: Base64-encoded string
    """
    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png', bbox_inches='tight')
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def plotly_fig_to_base64(fig) -> str:
    """
    Convert a plotly figure to base64 encoding for embedding in HTML.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        str: Base64-encoded string
    """
    img_bytes = io.BytesIO()
    fig.write_image(img_bytes, format='png', width=900, height=500)
    img_bytes.seek(0)
    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')
    return img_base64

def generate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for the dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dict[str, Any]: Dictionary with data summary
    """
    summary = {}
    
    # Basic info
    summary['n_rows'] = len(df)
    summary['n_columns'] = len(df.columns)
    
    # Data types
    dtype_counts = df.dtypes.value_counts().to_dict()
    summary['dtype_counts'] = {str(k): v for k, v in dtype_counts.items()}
    
    # Column-wise summary
    cols_summary = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isna().sum(),
            'missing_pct': 100 * df[col].isna().mean(),
        }
        
        # Add type-specific information
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'type_category': 'numeric'
            })
        elif pd.api.types.is_datetime64_dtype(df[col]):
            col_info.update({
                'min': str(df[col].min()),
                'max': str(df[col].max()),
                'type_category': 'datetime'
            })
        else:
            col_info.update({
                'unique_values': df[col].nunique(),
                'type_category': 'categorical'
            })
            
            # Add top categories if categorical with manageable number of values
            if df[col].nunique() <= 10:
                top_vals = df[col].value_counts().head(5).to_dict()
                col_info['top_values'] = {str(k): v for k, v in top_vals.items()}
        
        cols_summary.append(col_info)
    
    summary['columns'] = cols_summary
    
    return summary

def generate_cleaning_summary(original_df: pd.DataFrame, 
                             cleaned_df: pd.DataFrame,
                             cleaning_steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary of data cleaning steps.
    
    Args:
        original_df: Original dataframe
        cleaned_df: Cleaned dataframe
        cleaning_steps: List of cleaning steps performed
        
    Returns:
        Dict[str, Any]: Dictionary with cleaning summary
    """
    summary = {}
    
    # Compare shapes
    summary['original_shape'] = original_df.shape
    summary['cleaned_shape'] = cleaned_df.shape
    
    # Missing values before and after
    summary['original_missing'] = original_df.isna().sum().sum()
    summary['cleaned_missing'] = cleaned_df.isna().sum().sum()
    
    # Columns changed
    cols_before = set(original_df.columns)
    cols_after = set(cleaned_df.columns)
    summary['cols_added'] = list(cols_after - cols_before)
    summary['cols_removed'] = list(cols_before - cols_after)
    
    # Changes in data types
    dtype_changes = {}
    for col in cols_before.intersection(cols_after):
        if original_df[col].dtype != cleaned_df[col].dtype:
            dtype_changes[col] = {
                'before': str(original_df[col].dtype),
                'after': str(cleaned_df[col].dtype)
            }
    
    summary['dtype_changes'] = dtype_changes
    
    # Cleaning steps record
    summary['cleaning_steps'] = cleaning_steps
    
    return summary

def generate_basic_eda_summary(df: pd.DataFrame, 
                              visualizations: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary of basic EDA steps.
    
    Args:
        df: Input dataframe
        visualizations: Dictionary of visualizations generated
        
    Returns:
        Dict[str, Any]: Dictionary with EDA summary
    """
    summary = {}
    
    # Basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        corr_matrix = df[numeric_cols].corr().round(2).to_dict()
        summary['correlation'] = corr_matrix
    
    # Categorical column summaries
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    if cat_cols:
        cat_summaries = {}
        for col in cat_cols:
            if df[col].nunique() <= 30:  # Only for columns with reasonable number of categories
                cat_summaries[col] = df[col].value_counts().head(10).to_dict()
        
        summary['categorical_summaries'] = cat_summaries
    
    # Store visualizations
    summary['visualizations'] = visualizations
    
    return summary

def generate_advanced_eda_summary(df: pd.DataFrame, 
                                analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate summary of advanced EDA steps.
    
    Args:
        df: Input dataframe
        analysis_results: Dictionary of advanced analysis results
        
    Returns:
        Dict[str, Any]: Dictionary with advanced EDA summary
    """
    summary = {}
    
    # Store all analysis results
    summary.update(analysis_results)
    
    return summary

def generate_figure_caption(fig_type: str, 
                           data_info: Dict[str, Any], 
                           openai_key: Optional[str] = None) -> str:
    """
    Generate a caption for a figure using OpenAI API if available.
    
    Args:
        fig_type: Type of figure (histogram, scatter, etc.)
        data_info: Information about the data in the figure
        openai_key: OpenAI API key (optional)
        
    Returns:
        str: Generated caption
    """
    # Default captions if no API key or API call fails
    default_captions = {
        'histogram': f"Histogram showing the distribution of {data_info.get('variable', 'the variable')}.",
        'scatter': f"Scatter plot showing the relationship between {data_info.get('x_var', 'x')} and {data_info.get('y_var', 'y')}.",
        'boxplot': f"Box plot showing the distribution of {data_info.get('variable', 'the variable')} across different categories.",
        'correlation': f"Correlation matrix showing the relationships between numeric variables in the dataset.",
        'time_series': f"Time series plot showing the trend of {data_info.get('variable', 'the variable')} over time.",
        'bar': f"Bar chart showing the counts of {data_info.get('variable', 'the variable')}."
    }
    
    # If no OpenAI key, return default caption
    if not openai_key:
        return default_captions.get(fig_type, "Figure showing data visualization.")
    
    try:
        # Configure OpenAI
        openai.api_key = openai_key
        
        # Create prompt based on figure type and data info
        prompt = f"Write a short, informative one-sentence caption for a {fig_type} chart "
        
        if fig_type == 'histogram':
            prompt += f"showing the distribution of {data_info.get('variable')}. "
            if 'stats' in data_info:
                stats = data_info['stats']
                prompt += f"The mean is {stats.get('mean'):.2f}, the median is {stats.get('median'):.2f}, "
                prompt += f"and the standard deviation is {stats.get('std'):.2f}."
                
        elif fig_type == 'scatter':
            prompt += f"showing the relationship between {data_info.get('x_var')} and {data_info.get('y_var')}. "
            if 'correlation' in data_info:
                prompt += f"The correlation is {data_info.get('correlation'):.2f}."
                
        elif fig_type == 'boxplot':
            prompt += f"showing the distribution of {data_info.get('variable')} across different categories. "
            if 'outliers' in data_info:
                prompt += f"There are {data_info.get('outliers')} outliers."
                
        elif fig_type == 'correlation':
            prompt += f"showing the relationships between numeric variables in the dataset."
            
        elif fig_type == 'time_series':
            prompt += f"showing the trend of {data_info.get('variable')} over time."
            
        elif fig_type == 'bar':
            prompt += f"showing the counts of {data_info.get('variable')}."
        
        # Add instruction for style
        prompt += " The caption should be factual and highlight the most important insight."
        
        # Call OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose appropriate engine
            prompt=prompt,
            max_tokens=50,
            temperature=0.7,
            n=1
        )
        
        caption = response.choices[0].text.strip()
        
        # If caption is empty, return default
        if not caption:
            return default_captions.get(fig_type, "Figure showing data visualization.")
            
        return caption
        
    except Exception as e:
        print(f"Error generating caption with OpenAI: {str(e)}")
        # Fall back to default caption
        return default_captions.get(fig_type, "Figure showing data visualization.")

def generate_section_narrative(section_type: str, 
                              data: Dict[str, Any],
                              openai_key: Optional[str] = None) -> str:
    """
    Generate a narrative summary for a report section using OpenAI API if available.
    
    Args:
        section_type: Type of section (data_summary, cleaning, basic_eda, advanced_eda)
        data: Data for the section
        openai_key: OpenAI API key (optional)
        
    Returns:
        str: Generated narrative
    """
    # Default narratives if no API key or API call fails
    default_narratives = {
        'data_summary': "The dataset contains {n_rows} rows and {n_columns} columns. "
                      "The data includes numeric, categorical, and datetime features.",
        'cleaning': "Data cleaning involved handling missing values, converting data types, "
                  "and transforming features. {n_added} new features were created and "
                  "{n_removed} features were removed.",
        'basic_eda': "Exploratory data analysis revealed the distributions and relationships "
                   "between variables in the dataset.",
        'advanced_eda': "Advanced analysis included dimensionality reduction, clustering, "
                       "and hypothesis testing to uncover deeper patterns in the data."
    }
    
    # Fill in placeholders in default narratives
    if section_type == 'data_summary':
        default_narrative = default_narratives['data_summary'].format(
            n_rows=data.get('n_rows', 'N/A'),
            n_columns=data.get('n_columns', 'N/A')
        )
    elif section_type == 'cleaning':
        default_narrative = default_narratives['cleaning'].format(
            n_added=len(data.get('cols_added', [])),
            n_removed=len(data.get('cols_removed', []))
        )
    else:
        default_narrative = default_narratives.get(section_type, "This section provides analysis of the data.")
    
    # If no OpenAI key, return default narrative
    if not openai_key:
        return default_narrative
    
    try:
        # Configure OpenAI
        openai.api_key = openai_key
        
        # Create prompt based on section type and data
        prompt = f"Write a short, informative paragraph (3-5 sentences) summarizing the following "
        
        if section_type == 'data_summary':
            prompt += f"dataset: It has {data.get('n_rows')} rows and {data.get('n_columns')} columns. "
            prompt += f"The data types include {', '.join(data.get('dtype_counts', {}).keys())}. "
            
            # Add info about numeric columns
            numeric_cols = [col for col in data.get('columns', []) if col.get('type_category') == 'numeric']
            if numeric_cols:
                prompt += f"Numeric columns include {', '.join([col['name'] for col in numeric_cols[:3]])}. "
            
            # Add info about categorical columns
            cat_cols = [col for col in data.get('columns', []) if col.get('type_category') == 'categorical']
            if cat_cols:
                prompt += f"Categorical columns include {', '.join([col['name'] for col in cat_cols[:3]])}. "
                
        elif section_type == 'cleaning':
            prompt += f"data cleaning process: The original shape was {data.get('original_shape')} and "
            prompt += f"the cleaned shape is {data.get('cleaned_shape')}. "
            
            if data.get('cols_added'):
                prompt += f"New features created: {', '.join(data.get('cols_added'))}. "
                
            if data.get('cols_removed'):
                prompt += f"Features removed: {', '.join(data.get('cols_removed'))}. "
                
            prompt += f"There were {data.get('original_missing', 0)} missing values before cleaning "
            prompt += f"and {data.get('cleaned_missing', 0)} missing values after cleaning."
                
        elif section_type == 'basic_eda':
            prompt += "exploratory data analysis: "
            
            if 'numeric_stats' in data:
                prompt += "The analysis includes descriptive statistics for numeric variables. "
                
            if 'correlation' in data:
                prompt += "Correlation analysis was performed between numeric variables. "
                
            if 'categorical_summaries' in data:
                prompt += "Categorical variable distributions were analyzed. "
                
            if 'visualizations' in data:
                vis_types = list(data.get('visualizations', {}).keys())
                if vis_types:
                    prompt += f"Visualizations include {', '.join(vis_types)}."
                
        elif section_type == 'advanced_eda':
            prompt += "advanced exploratory data analysis: "
            
            analyses_performed = []
            if 'pca_results' in data:
                analyses_performed.append("Principal Component Analysis")
                
            if 'tsne_results' in data:
                analyses_performed.append("t-SNE dimensionality reduction")
                
            if 'umap_results' in data:
                analyses_performed.append("UMAP dimensionality reduction")
                
            if 'clustering_results' in data:
                analyses_performed.append("Clustering analysis")
                
            if 'hypothesis_tests' in data:
                analyses_performed.append("Hypothesis testing")
                
            if analyses_performed:
                prompt += f"Analyses performed include {', '.join(analyses_performed)}."
        
        # Add instruction for style
        prompt += " The paragraph should be factual, informative, and highlight key insights. Do not use bullet points."
        
        # Call OpenAI API
        response = openai.Completion.create(
            engine="text-davinci-003",  # Choose appropriate engine
            prompt=prompt,
            max_tokens=150,
            temperature=0.7,
            n=1
        )
        
        narrative = response.choices[0].text.strip()
        
        # If narrative is empty, return default
        if not narrative:
            return default_narrative
            
        return narrative
        
    except Exception as e:
        print(f"Error generating narrative with OpenAI: {str(e)}")
        # Fall back to default narrative
        return default_narrative

def generate_html_report(
    df: pd.DataFrame,
    config: Dict[str, Any],
    data_summary: Dict[str, Any],
    cleaning_summary: Optional[Dict[str, Any]] = None,
    basic_eda_summary: Optional[Dict[str, Any]] = None,
    advanced_eda_summary: Optional[Dict[str, Any]] = None,
    openai_key: Optional[str] = None
) -> str:
    """
    Generate an HTML report.
    
    Args:
        df: Input dataframe
        config: Report configuration
        data_summary: Data summary information
        cleaning_summary: Data cleaning summary
        basic_eda_summary: Basic EDA summary
        advanced_eda_summary: Advanced EDA summary
        openai_key: OpenAI API key for generating narratives (optional)
        
    Returns:
        str: HTML report content
    """
    # Define HTML template with Bootstrap for styling
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ config.title }}</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; }
            h1, h2, h3 { color: #2C3E50; }
            .table-responsive { margin: 20px 0; }
            .figure { margin: 20px 0; text-align: center; }
            .figure img { max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }
            .figure-caption { font-style: italic; color: #666; margin-top: 8px; }
            .section-narrative { background-color: #f8f9fa; padding: 15px; border-radius: 8px; 
                               border-left: 4px solid #4e73df; margin-bottom: 20px; }
            .footer { margin-top: 50px; padding-top: 20px; border-top: 1px solid #eee; color: #666; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="page-header">
                <h1>{{ config.title }}</h1>
                <p>Generated on {{ config.date }} by {{ config.author }}</p>
                <hr>
            </div>
            
            <div class="table-of-contents mb-4">
                <h3>Table of Contents</h3>
                <ol>
                    <li><a href="#data-summary">Data Summary</a></li>
                    {% if config.include_cleaning_summary %}
                    <li><a href="#data-cleaning">Data Cleaning</a></li>
                    {% endif %}
                    {% if config.include_basic_eda %}
                    <li><a href="#basic-eda">Basic Exploratory Analysis</a></li>
                    {% endif %}
                    {% if config.include_advanced_eda %}
                    <li><a href="#advanced-eda">Advanced Analysis</a></li>
                    {% endif %}
                </ol>
            </div>
            
            <!-- Data Summary Section -->
            <section id="data-summary" class="mb-5">
                <h2>Data Summary</h2>
                {% if config.auto_generate_text %}
                <div class="section-narrative">
                    <p>{{ data_summary_narrative }}</p>
                </div>
                {% endif %}
                
                <h3>Dataset Overview</h3>
                <table class="table table-bordered">
                    <tr><th>Number of Rows</th><td>{{ data_summary.n_rows }}</td></tr>
                    <tr><th>Number of Columns</th><td>{{ data_summary.n_columns }}</td></tr>
                    <tr>
                        <th>Data Types</th>
                        <td>
                            {% for dtype, count in data_summary.dtype_counts.items() %}
                            {{ dtype }}: {{ count }}{% if not loop.last %}, {% endif %}
                            {% endfor %}
                        </td>
                    </tr>
                </table>
                
                <h3>Column Information</h3>
                <div class="table-responsive">
                    <table class="table table-striped table-sm">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Data Type</th>
                                <th>Missing</th>
                                <th>Missing %</th>
                                <th>Additional Info</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col in data_summary.columns %}
                            <tr>
                                <td>{{ col.name }}</td>
                                <td>{{ col.dtype }}</td>
                                <td>{{ col.missing_count }}</td>
                                <td>{{ "%.2f"|format(col.missing_pct) }}%</td>
                                <td>
                                    {% if col.type_category == 'numeric' %}
                                    Range: {{ "%.2f"|format(col.min) }} to {{ "%.2f"|format(col.max) }}, 
                                    Mean: {{ "%.2f"|format(col.mean) }}, 
                                    StdDev: {{ "%.2f"|format(col.std) }}
                                    {% elif col.type_category == 'categorical' %}
                                    {{ col.unique_values }} unique values
                                    {% if col.top_values %}
                                    <br>Top values: 
                                    {% for val, count in col.top_values.items() %}
                                    {{ val }} ({{ count }}){% if not loop.last %}, {% endif %}
                                    {% endfor %}
                                    {% endif %}
                                    {% elif col.type_category == 'datetime' %}
                                    Range: {{ col.min }} to {{ col.max }}
                                    {% endif %}
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </section>
            
            <!-- Data Cleaning Section -->
            {% if config.include_cleaning_summary and cleaning_summary %}
            <section id="data-cleaning" class="mb-5">
                <h2>Data Cleaning</h2>
                {% if config.auto_generate_text %}
                <div class="section-narrative">
                    <p>{{ cleaning_summary_narrative }}</p>
                </div>
                {% endif %}
                
                <h3>Changes Overview</h3>
                <table class="table table-bordered">
                    <tr>
                        <th>Original Shape</th>
                        <td>{{ cleaning_summary.original_shape[0] }} rows × {{ cleaning_summary.original_shape[1] }} columns</td>
                    </tr>
                    <tr>
                        <th>Cleaned Shape</th>
                        <td>{{ cleaning_summary.cleaned_shape[0] }} rows × {{ cleaning_summary.cleaned_shape[1] }} columns</td>
                    </tr>
                    <tr>
                        <th>Missing Values (Before)</th>
                        <td>{{ cleaning_summary.original_missing }}</td>
                    </tr>
                    <tr>
                        <th>Missing Values (After)</th>
                        <td>{{ cleaning_summary.cleaned_missing }}</td>
                    </tr>
                </table>
                
                {% if cleaning_summary.cols_added %}
                <h3>Columns Added</h3>
                <ul>
                    {% for col in cleaning_summary.cols_added %}
                    <li>{{ col }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if cleaning_summary.cols_removed %}
                <h3>Columns Removed</h3>
                <ul>
                    {% for col in cleaning_summary.cols_removed %}
                    <li>{{ col }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if cleaning_summary.dtype_changes %}
                <h3>Data Type Changes</h3>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Column</th>
                                <th>Original Type</th>
                                <th>New Type</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for col, change in cleaning_summary.dtype_changes.items() %}
                            <tr>
                                <td>{{ col }}</td>
                                <td>{{ change.before }}</td>
                                <td>{{ change.after }}</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
                
                {% if cleaning_summary.cleaning_steps %}
                <h3>Cleaning Steps</h3>
                <ol>
                    {% for step in cleaning_summary.cleaning_steps %}
                    <li>
                        <strong>{{ step.action }}</strong>: {{ step.description }}
                    </li>
                    {% endfor %}
                </ol>
                {% endif %}
            </section>
            {% endif %}
            
            <!-- Basic EDA Section -->
            {% if config.include_basic_eda and basic_eda_summary %}
            <section id="basic-eda" class="mb-5">
                <h2>Basic Exploratory Analysis</h2>
                {% if config.auto_generate_text %}
                <div class="section-narrative">
                    <p>{{ basic_eda_narrative }}</p>
                </div>
                {% endif %}
                
                {% if basic_eda_summary.visualizations %}
                <h3>Visualizations</h3>
                {% for viz_name, viz_data in basic_eda_summary.visualizations.items() %}
                {% if viz_name in config.selected_visualizations or not config.selected_visualizations %}
                <div class="figure">
                    <img src="data:image/png;base64,{{ viz_data.image }}" alt="{{ viz_name }}">
                    <div class="figure-caption">{{ viz_data.caption }}</div>
                </div>
                <hr>
                {% endif %}
                {% endfor %}
                {% endif %}
                
                {% if basic_eda_summary.numeric_stats %}
                <h3>Descriptive Statistics</h3>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Statistic</th>
                                {% for column in basic_eda_summary.numeric_stats.keys() %}
                                <th>{{ column }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for stat in ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'] %}
                            <tr>
                                <td><strong>{{ stat }}</strong></td>
                                {% for column in basic_eda_summary.numeric_stats.keys() %}
                                <td>{{ "%.2f"|format(basic_eda_summary.numeric_stats[column][stat]) }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
                
                {% if basic_eda_summary.correlation %}
                <h3>Correlation Matrix</h3>
                <div class="table-responsive">
                    <table class="table table-bordered table-sm">
                        <thead>
                            <tr>
                                <th></th>
                                {% for column in basic_eda_summary.correlation.keys() %}
                                <th>{{ column }}</th>
                                {% endfor %}
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in basic_eda_summary.correlation.keys() %}
                            <tr>
                                <td><strong>{{ row }}</strong></td>
                                {% for column in basic_eda_summary.correlation.keys() %}
                                <td>{{ "%.2f"|format(basic_eda_summary.correlation[row][column]) }}</td>
                                {% endfor %}
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
            </section>
            {% endif %}
            
            <!-- Advanced EDA Section -->
            {% if config.include_advanced_eda and advanced_eda_summary %}
            <section id="advanced-eda" class="mb-5">
                <h2>Advanced Analysis</h2>
                {% if config.auto_generate_text %}
                <div class="section-narrative">
                    <p>{{ advanced_eda_narrative }}</p>
                </div>
                {% endif %}
                
                {% if advanced_eda_summary.visualizations %}
                <h3>Advanced Visualizations</h3>
                {% for viz_name, viz_data in advanced_eda_summary.visualizations.items() %}
                {% if viz_name in config.selected_visualizations or not config.selected_visualizations %}
                <div class="figure">
                    <img src="data:image/png;base64,{{ viz_data.image }}" alt="{{ viz_name }}">
                    <div class="figure-caption">{{ viz_data.caption }}</div>
                </div>
                <hr>
                {% endif %}
                {% endfor %}
                {% endif %}
                
                {% if advanced_eda_summary.pca_summary %}
                <h3>Principal Component Analysis</h3>
                <table class="table table-bordered">
                    <tr>
                        <th>Number of Components</th>
                        <td>{{ advanced_eda_summary.pca_summary.n_components }}</td>
                    </tr>
                    <tr>
                        <th>Cumulative Explained Variance</th>
                        <td>{{ "%.2f"|format(advanced_eda_summary.pca_summary.cumulative_var * 100) }}%</td>
                    </tr>
                </table>
                {% endif %}
                
                {% if advanced_eda_summary.clustering_summary %}
                <h3>Clustering Analysis</h3>
                <table class="table table-bordered">
                    <tr>
                        <th>Method</th>
                        <td>{{ advanced_eda_summary.clustering_summary.method }}</td>
                    </tr>
                    <tr>
                        <th>Number of Clusters</th>
                        <td>{{ advanced_eda_summary.clustering_summary.n_clusters }}</td>
                    </tr>
                </table>
                {% if advanced_eda_summary.clustering_summary.cluster_sizes %}
                <h4>Cluster Sizes</h4>
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Cluster</th>
                                <th>Size</th>
                                <th>Percentage</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for cluster, size in advanced_eda_summary.clustering_summary.cluster_sizes.items() %}
                            <tr>
                                <td>{{ cluster }}</td>
                                <td>{{ size.count }}</td>
                                <td>{{ "%.2f"|format(size.percentage) }}%</td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
                {% endif %}
                {% endif %}
                
                {% if advanced_eda_summary.hypothesis_tests %}
                <h3>Hypothesis Tests</h3>
                {% for test in advanced_eda_summary.hypothesis_tests %}
                <div class="card mb-4">
                    <div class="card-header">
                        <h4>{{ test.test_type }}</h4>
                    </div>
                    <div class="card-body">
                        <p><strong>Variables:</strong> {{ test.variables }}</p>
                        <p><strong>P-value:</strong> {{ "%.4f"|format(test.p_value) }}</p>
                        <p><strong>Result:</strong> {{ test.result }}</p>
                        {% if test.effect_size %}
                        <p><strong>Effect Size:</strong> {{ "%.4f"|format(test.effect_size) }} ({{ test.effect_size_interpretation }})</p>
                        {% endif %}
                    </div>
                </div>
                {% endfor %}
                {% endif %}
            </section>
            {% endif %}
            
            <div class="footer">
                <p>Generated by Streamlit EDA App</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Generate narratives if auto-generate is enabled
    data_summary_narrative = ""
    cleaning_summary_narrative = ""
    basic_eda_narrative = ""
    advanced_eda_narrative = ""
    
    if config['auto_generate_text']:
        data_summary_narrative = generate_section_narrative('data_summary', data_summary, openai_key)
        
        if config['include_cleaning_summary'] and cleaning_summary:
            cleaning_summary_narrative = generate_section_narrative('cleaning', cleaning_summary, openai_key)
            
        if config['include_basic_eda'] and basic_eda_summary:
            basic_eda_narrative = generate_section_narrative('basic_eda', basic_eda_summary, openai_key)
            
        if config['include_advanced_eda'] and advanced_eda_summary:
            advanced_eda_narrative = generate_section_narrative('advanced_eda', advanced_eda_summary, openai_key)
    
    # Render the template
    template = jinja2.Template(html_template)
    html_content = template.render(
        config=config,
        data_summary=data_summary,
        cleaning_summary=cleaning_summary,
        basic_eda_summary=basic_eda_summary,
        advanced_eda_summary=advanced_eda_summary,
        data_summary_narrative=data_summary_narrative,
        cleaning_summary_narrative=cleaning_summary_narrative,
        basic_eda_narrative=basic_eda_narrative,
        advanced_eda_narrative=advanced_eda_narrative
    )
    
    return html_content

def generate_pdf_from_html(html_content: str) -> bytes:
    """
    Generate a PDF from HTML content.
    
    This function is deprecated and will raise a NotImplementedError since PDF
    generation has been temporarily disabled due to GTK dependency issues.
    
    Args:
        html_content: HTML content to convert
        
    Returns:
        bytes: PDF file content
        
    Raises:
        NotImplementedError: This functionality is currently disabled
    """
    raise NotImplementedError(
        "PDF generation is temporarily disabled due to system dependencies. "
        "HTML reports are provided instead, which can be printed to PDF using your browser."
    )
    
    # The following code is kept for future reference but not executed
    """
    try:
        import weasyprint
        pdf = weasyprint.HTML(string=html_content).write_pdf()
        return pdf
    except OSError as e:
        if "cannot load library 'gobject-2.0-0'" in str(e):
            error_msg = (
                "GTK libraries required for PDF generation are not properly installed or configured."
            )
            raise RuntimeError(error_msg) from e
        else:
            raise
    """

def save_session_state(
    session_data: Dict[str, Any],
    filename: str = "eda_session.pkl"
) -> bool:
    """
    Save session state to file.
    
    Args:
        session_data: Session data to save
        filename: Output filename
        
    Returns:
        bool: Success or failure
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(session_data, f)
        return True
    except Exception as e:
        print(f"Error saving session state: {str(e)}")
        return False

def load_session_state(filename: str = "eda_session.pkl") -> Optional[Dict[str, Any]]:
    """
    Load session state from file.
    
    Args:
        filename: Input filename
        
    Returns:
        Optional[Dict[str, Any]]: Loaded session data, None if error
    """
    try:
        with open(filename, 'rb') as f:
            session_data = pickle.load(f)
        return session_data
    except Exception as e:
        print(f"Error loading session state: {str(e)}")
        return None

def report_generation_ui(
    df: pd.DataFrame, 
    cleaning_steps: List[Dict[str, Any]],
    basic_eda_results: Dict[str, Any],
    advanced_eda_results: Dict[str, Any]
) -> None:
    """
    Streamlit UI for report generation.
    
    Args:
        df: Input cleaned dataframe
        cleaning_steps: List of cleaning steps performed
        basic_eda_results: Basic EDA results
        advanced_eda_results: Advanced EDA results
    """
    st.title("Report Generation")
    
    if df is None or df.empty:
        st.warning("Please upload, clean, and analyze a dataset first.")
        return
    
    # Get original dataframe if available in session state
    original_df = st.session_state.get('original_df', df)
    
    # Report configuration
    st.subheader("Report Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        title = st.text_input("Report Title", value="EDA Report")
        author = st.text_input("Author", value="Streamlit EDA App")
    
    with col2:
        include_data_summary = st.checkbox("Include Data Summary", value=True)
        include_cleaning = st.checkbox("Include Data Cleaning Summary", value=True)
        include_basic_eda = st.checkbox("Include Basic EDA", value=True)
        include_advanced_eda = st.checkbox("Include Advanced EDA", value=True)
    
    # OpenAI API key for captions and narratives
    use_ai_text = st.checkbox("Generate AI Text Explanations", value=False)
    
    if use_ai_text:
        openai_key = st.text_input("OpenAI API Key", type="password")
        if not openai_key:
            st.warning("Please enter an OpenAI API key to enable AI-generated text. If left blank, default text will be used.")
    else:
        openai_key = None
    
    # Select visualizations
    st.subheader("Select Visualizations to Include")
    
    all_visualizations = {}
    
    # Collect all visualizations
    if include_basic_eda and 'visualizations' in basic_eda_results:
        all_visualizations.update({f"Basic: {k}": k for k in basic_eda_results['visualizations'].keys()})
    
    if include_advanced_eda and 'visualizations' in advanced_eda_results:
        all_visualizations.update({f"Advanced: {k}": k for k in advanced_eda_results['visualizations'].keys()})
    
    if all_visualizations:
        selected_viz = st.multiselect(
            "Select visualizations to include in the report:",
            options=list(all_visualizations.keys()),
            default=list(all_visualizations.keys())
        )
        
        selected_visualizations = [all_visualizations[viz] for viz in selected_viz]
    else:
        selected_visualizations = []
    
    # Create report configuration
    report_config = create_report_config(
        title=title,
        author=author,
        include_data_summary=include_data_summary,
        include_cleaning_summary=include_cleaning,
        include_basic_eda=include_basic_eda,
        include_advanced_eda=include_advanced_eda,
        selected_visualizations=selected_visualizations,
        auto_generate_text=use_ai_text
    )
    
    # Generate report button
    if st.button("Generate Report"):
        with st.spinner("Generating report..."):
            # Prepare summaries
            data_summary = generate_data_summary(df)
            
            cleaning_summary = None
            if include_cleaning:
                cleaning_summary = generate_cleaning_summary(original_df, df, cleaning_steps)
            
            basic_eda_summary = None
            if include_basic_eda:
                basic_eda_summary = generate_basic_eda_summary(df, basic_eda_results)
            
            advanced_eda_summary = None
            if include_advanced_eda:
                advanced_eda_summary = generate_advanced_eda_summary(df, advanced_eda_results)
              # Generate HTML report
            html_content = generate_html_report(
                df=df,
                config=report_config,
                data_summary=data_summary,
                cleaning_summary=cleaning_summary,
                basic_eda_summary=basic_eda_summary,
                advanced_eda_summary=advanced_eda_summary,
                openai_key=openai_key
            )
                
            # Provide download option for HTML report
            st.success("Report generated successfully!")
            
            # Display note about PDF deprecation
            st.info("⚠️ NOTE: PDF report generation has been temporarily deprecated due to system dependencies. "
                   "See the README for more information. HTML reports provide the same content and can be "
                   "printed to PDF using your browser.")
            
            st.download_button(
                label="Download HTML Report",
                data=html_content,
                file_name=f"eda_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )
    
    # Session management
    st.subheader("Session Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Save session
        save_filename = st.text_input("Session filename", value="eda_session.pkl")
        
        if st.button("Save Session"):
            # Prepare session data
            session_data = {
                'df': df,
                'original_df': original_df,
                'cleaning_steps': cleaning_steps,
                'basic_eda_results': basic_eda_results,
                'advanced_eda_results': advanced_eda_results,
                'report_config': report_config
            }
            
            success = save_session_state(session_data, save_filename)
            
            if success:
                st.success(f"Session saved to {save_filename}")
            else:
                st.error("Error saving session")
    
    with col2:
        # Load session
        uploaded_file = st.file_uploader("Load session file", type=['pkl'])
        
        if uploaded_file is not None:
            try:
                # Save uploaded file temporarily
                with open("temp_session.pkl", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Load session data
                session_data = load_session_state("temp_session.pkl")
                
                if session_data:
                    # Update session state
                    st.session_state['df'] = session_data['df']
                    st.session_state['original_df'] = session_data['original_df']
                    st.session_state['cleaning_steps'] = session_data['cleaning_steps']
                    st.session_state['basic_eda_results'] = session_data['basic_eda_results']
                    st.session_state['advanced_eda_results'] = session_data['advanced_eda_results']
                    
                    st.success("Session loaded successfully! Please refresh the page to see the loaded data.")
                else:
                    st.error("Error loading session data")
                
                # Clean up temporary file
                if os.path.exists("temp_session.pkl"):
                    os.remove("temp_session.pkl")
                    
            except Exception as e:
                st.error(f"Error loading session: {str(e)}")
                
                # Clean up temporary file
                if os.path.exists("temp_session.pkl"):
                    os.remove("temp_session.pkl")
