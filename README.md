# General Purpose EDA App

A streamlined Exploratory Data Analysis (EDA) application built with Streamlit that enables users to upload, clean, analyze, and generate insights from datasets with an intuitive interface. The app was developed by agentic AI (i.e., Claude Sonnet 3.7) by prompting our [requirements document](https://docs.google.com/document/d/1maDOPo9EgSe0kNaPlSoFTTMTx5SKCdyMNbW4fnoQ6CY/edit?usp=sharing). This document was collaboratively developed and represents one of the main efforts of this project, on par with  refining the apps functionality from the AI-generated starting point

## Features

### Implemented

- **Data Upload**: own data in CSV & Excel formats or choose from sample data
- **Data Cleaning**: Missing value detection, adaptation of missing values, variables type conversion, normalization, feature selection 
- **Basic EDA**: Descriptive statistics, distribution visualization, outlier detection, relationship analysis, time series analysis
- **Advanced EDA**: Correlation matrices, dimensionality reduction (PCA, t-SNE, UMAP), clustering, hypothesis testing

### Outlook

- **Report Generation**: Customizable HTML/PDF reports with visualizations and insights summary
- **Save Session State**: Ability to save analysis progress and reload for continued work
- **AI-Assisted Insights**: Integration with LLMs for automated data interpretation
- **Advanced Modeling**: Construction and trainig of ML-models with analysed data

## For Users

A webhooked version of the app is available at: https://ai-code-project-eda-app.streamlit.app/

Note: The app might need to be woken up on first access.

## For Developers

### Using pip

```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

### Using conda (recommended)

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate eda_app

# Run application
streamlit run app.py
```

## Open Source

This project is licensed under the MIT License.

### Sample Datasets

* [Billionaires Statistics Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset)
* [Titanic Dataset](https://www.kaggle.com/c/titanic/)
* [Student Academic Performance Dataset](https://www.kaggle.com/datasets/firedmosquito831/student-academic-performance-simulation-4000)
