# App Requirements Document

# 1\. Introduction

This document specifies the requirements for a general EDA app. It includes functional and non-functional requirements, as well as use cases.

Our General-Purpose EDA Streamlit App should be a modular, high-performance, and scientifically rigorous exploratory data analysis (EDA) tool built in Python using Streamlit. Designed for researchers, data scientists, and analysts, the app enables quick data profiling, cleaning, transformation, and advanced statistical insights — all within an interactive browser-based interface.

Features:

* **Flexible Data Upload**: Supports `.csv`, `.xlsx`, `.xls` formats up to 5GB. Validates file type, structure, and content (requires at least one numeric column).  
* **Cleaning & Filtering**: Detect and correct non-standard missing values, normalize numeric columns, convert data types, and create custom features.  
* **Basic EDA**: Visualize feature distributions, outliers, and data types; compute descriptive statistics; explore variable relationships and target interactions.  
* **Advanced EDA**: Perform dimensionality reduction (PCA, t-SNE, UMAP), clustering (KMeans, DBSCAN), correlation analysis (with p-values), and hypothesis testing (t-tests, ANOVA).  
* **Time Series Support**: Automatically detects datetime columns and enables time-based grouping and visualization.  
* **Export & Reporting**: Generate customizable HTML/PDF reports with optional AI-generated summaries and captions. Save session states for later reuse.

Design Principles:

* **Reusability**: Fully modular with function-based structure. All functions are reusable, testable, and single-responsibility.  
* **Readability**: Clean, PEP8-compliant code with type hints and in-line documentation.  
* **Reproducibility**: Randomness controlled with fixed seeds. No hardcoded file paths.  
* **Performance**: Optimized for vectorized operations using Pandas/NumPy. Avoids row-wise iteration and unnecessary apply-based logic.  
* **Data-Agnostic**: Designed to work with diverse datasets (no fixed schema), using intelligent type inference and flexible logic paths.

Tech Stack

* **Frontend**: [Streamlit](https://streamlit.io/)  
* **Backend/Data**: `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `scipy`, `plotly`

# 2\. Goals

The general purpose EDA should enable the user to upload a dataset and perform some cleaning and filtering for later analysis. This analysis included a basic EDA (i.e., histograms, descriptive statistics and basic visualizations) and an advanced one (unsupervised ML, correlations and hypothesis testing). Finally, the analysis can be exported into a customizable report.   
The app should be focused on scientific standards and implemented after the following best practices: 

\#\#\# Readability  
Code adheres to Python standards (PEP 8), uses clear variable names, proper indentation, in-line comments, and type hints where necessary.

\*\*Type hints example:\*\*   
\`\`\`python  
from typing import List, Dict, Tuple, Optional

age: int \= 25  
def greet(name: str) \-\> str:  
    return f"Hello, {name}\!"

def process\_data(items: List\[int\], config: Optional\[Dict\[str, str\]\] \= None) \-\> Tuple\[int, float\]:  
    \# Function implementation  
    return total, average  
\`\`\`

\#\#\# Reproducibility  
Code produces the same results across different runs, avoids hard-coded paths, and ensures randomness is controlled.

\*\*Path example:\*\*  
\`\`\`python  
from pathlib import Path  
import random  
import numpy as np

\# For file paths  
BASE\_DIR \= Path(\_\_file\_\_).resolve().parent.parent  
DATA\_PATH \= BASE\_DIR / "data" / "input.csv"

\# For controlling randomness  
random.seed(42)  
np.random.seed(42)  
\`\`\`

\#\#\# Reusability  
Code is modular, functions are well-structured with single responsibilities, and can be reused in other projects. No classes, only functions should be used.

\*\*Example of modular code:\*\*  
\`\`\`python  
def validate\_input(data: dict) \-\> bool:  
    """Validate the input data structure."""  
    \# Implementation  
      
def process\_data(data: dict) \-\> dict:  
    """Process the validated data."""  
    \# Implementation  
      
def main(input\_file: str) \-\> None:  
    """Main workflow function."""  
    data \= load\_data(input\_file)  
    if validate\_input(data):  
        result \= process\_data(data)  
        save\_results(result)  
\`\`\`

\#\#\# Data structuring 

Code uses appropriate built-in data structures (`list`, `tuple`, `set`, `dict`) based on problem requirements. Choices are guided by access patterns, performance and memory considerations, and semantic meaning (e.g., mutability, uniqueness, order). Data structures are consistently used and easy to interpret in the context of EDA operations.

\*\*Example of data structure code:\*\*  
\`\`\`\`python  
from typing import List, Dict, Set, Tuple

\# List used for ordered collections (e.g., column names)  
column\_names: List\[str\] \= \["age", "income", "gender"\]

\# Tuple used for immutable, fixed-size structures (e.g., column schema)  
column\_schema: List\[Tuple\[str, str\]\] \= \[("age", "int"), ("income", "float")\]

\# Set used for fast uniqueness checks (e.g., identifying categorical columns)  
categorical\_columns: Set\[str\] \= {"gender", "region"}

\# Dict used for mapping column names to summary statistics  
summary\_stats: Dict\[str, Dict\[str, float\]\] \= {  
    "age": {"mean": 34.2, "std": 5.3},  
    "income": {"mean": 58000, "median": 55000}  
}

\# Dict for storing grouped data (e.g., by region or category)  
grouped\_data: Dict\[str, List\[Dict\[str, float\]\]\] \= {  
    "North": \[{"age": 30, "income": 62000}, {"age": 40, "income": 68000}\],  
    "South": \[{"age": 25, "income": 50000}\]  
}

\# Implementation  
\`\`\`\`

\#\#\# Vectorization Over Row-wise Operations

Use vectorized operations (e.g., direct column math or NumPy functions) instead of `df.apply()` or row-wise iteration for better performance and readability.

\*\*Example of vectorized code:\*\*

\`\`\`\`python

def compute\_bmi(df: pd.DataFrame) \-\> pd.DataFrame:

    """Compute BMI using vectorized operations."""

    df\['bmi'\] \= df\['weight'\] / (df\['height'\] / 100\) \*\* 2

    return df

\# Implementation  
\`\`\`\`

# 3\. Functional Requirements

The following table outlines the functional requirements.  
**Legend**  
GLOB: Global Feature  
Upload: UPLD  
Filtering and Cleaning: ClFi  
Basic EDA: bEDA  
Advanced EDA: aEDA  
Report Generation: RGen

| Requirement ID | Description | User Story | Expected Behaviour/ Outcome |
| :---- | :---- | :---- | :---- |
| GLOB01 | Cached settings | User can save analysis to be revisit it next time when launching app. | Analysis is somehow saved in real time to be real loaded next time app is started. |
| UPLD01 | Upload dataset | As a user, I want to upload my dataset to the app.  | The app should have an upload button. When the user clicks on it, the user is then prompted to either drag and drop a file, or directly select from its respective folder. |
| UPLD02 | Upload dataset | As a user, I want to receive a notification if the upload was successful or not.  | User gets error message if dataset \>5 GB. User gets error message if file is not in either .csv or .xlsx or .xls format. User gets error message if dataset contains no more than one column with numerical data. If the uploaded file does not violate the mentioned conditions, the user gets a notification that the file upload was successful and that the process of analysing the data can start.  |
| ClFi01 | Cleaning and Filtering | As a user, I want to make sure that the missing data of each of my variables is correctly specified as such. | Each column of the data is analyzed for missing values that are not reported as such, i.e. as “NA”. Examples are symbols like points and dashes, or blank spaces. These are replaced with NA.   |
| ClFi02 | Cleaning and Filtering | As a user, I want to then decide if the missing values should stay reported as NAs, if they should be replaced by a value of my choice, or if the nearest neighbor value should be used.  | The app displays missing values and offers options to: leave them as NAs, replace them with a user-defined value, or impute using the nearest neighbor method. The selected option is applied to the dataset preview. |
| ClFi03 | Cleaning and Filtering | As a user, I want to make sure that numeric variables are properly coded as such, and categorical variables are coded as categorical.  | Each column of the data is screened and checked if it is numerical or categorical data. The app properly codes the variables as such, if they are not already stored this way.  |
| ClFi04 | Cleaning and Filtering | As a user, I want to have the option to normalize numeric variables so that they are on a comparable scale before optional modelling.  | The app provides optional normalization methods (e.g., min-max scaling, z-score standardization). If selected, the transformation is applied to a preview of the dataset. The original dataset remains unchanged unless the user confirms and applies the transformation. |
| ClFi05 | Cleaning and Filtering | As a user, I want to have the option to create new features from existing ones to potentially improve model performance. | The app allows users to optionally define new features using basic operations (e.g., addition, ratios, or transformations) on existing variables. The new features are added to a preview version of the dataset. The original uploaded dataset remains unchanged unless the user confirms and applies the changes. |
| ClFi06 | Cleaning and Filtering | As a user, I want the option to select which variables to include in the modelling step so I can control the model inputs. | The app displays all available variables and allows users to optionally include or exclude them via checkboxes. The selection is applied only to the dataset preview used in modelling. The original uploaded dataset is not modified unless the user chooses to save the selected subset. |
| ClFi07 | Cleaning and Filtering | As a user, I want to have the option to download the filtered and cleaned dataset at any point going forward.  | Once the filtering and cleaning operations have been performed, the app provides an option in the user interface to download the modified version of the dataset. This option remains available at all subsequent stages of the workflow. The downloaded file reflects all transformations applied up to that point. |
| bEDA01 | Data Type Summary | As a user, I want to understand the types of features in my dataset. | Display a summary of columns by data type: numerical, categorical, boolean, datetime, etc. Optionally allow reclassification or casting if misdetected. |
| bEDA02 | Feature Distributions | As a user, I want to explore the distribution of each feature so I can detect skewness and outliers. | Generate histograms and KDE plots for numerical features and bar plots for categorical ones. Allow options for bin size, log scale, and overlay of summary stats like mean and median. |
| bEDA03 | Descriptive Statistics Summary | As a user, I want to see summary statistics to understand central tendency and variability. | Display mean, median, mode, std dev, range, min/max, and percentiles (25th, 75th) in a sortable table. Allow grouping by categorical variables and filtering of columns. |
| bEDA04 | Outlier Detection with Boxplots | As a user, I want to detect outliers so I can decide how to handle them. | Show boxplots for numerical features, optionally grouped by categories. Highlight outlier points based on IQR or Z-score thresholds. Include options to export or inspect outlier rows. |
| bEDA05 | Feature Relationships | As a user, I want to see how variables interact to discover trends and dependencies. | Use scatter plots for numerical relationships, bar/violin/strip plots for categorical or mixed types. Enable grouping and coloring by category. Include trendline and correlation coefficient display. |
| bEDA06 | Target Variable Analysis | As a user, I want to understand how features relate to my target variable. | For classification: show distribution of features per class (boxplots, violin plots). For regression: show scatter plots and correlation. Include options to stratify, group, or color by target. |
| bEDA07 | Date/Time Feature Exploration | As a user, I want to understand trends in time-based data. | Automatically detect datetime columns and allow time-based grouping (by day, month, year). Display time series plots, seasonal trends, or cyclic patterns. |
| aEDA01 | Correlation Matrix | As a user, I want to see how features correlate so I can detect multicollinearity and strong associations. | Display a heatmap of Pearson/Spearman/Kendall correlations with options to choose method and filter features to apply function to. The heatmap should also display p-value for the respective correlation. |
| aEDA02 | Dimensionality Reduction (PCA, t-SNE, UMAP) | As a user, I want to reduce my dataset to 2D to visualize complex patterns. | Generate interactive 2D scatterplots based on PCA, t-SNE and UMAP. Include settings for clustering methods such as which PCs to display for PCA, or hyper parameters of t-SNE/UMAP, as well as options for coloring by class or clusters. |
| aEDA03 | Clustering (KMeans, DBSCAN) | As a user, I want to identify latent groups within my data, and control different hyperparameters of these methods. | Let users apply clustering methods and visualize results on a 2D plot. Display summary stats per cluster. |
| aEDA04 | Hypothesis Testing (e.g. t-test, ANOVA) | As a user, I want to test whether group differences are statistically significant. | Select variables and groups to test. Output test result (p-value, effect size) and assumptions check. |
| aEDA06 | Correlation vs. Causation Helper | As a user, I want guidance on whether observed correlations might be confounded. | App suggests confounding checks or conditional plots (e.g. stratified scatter plots). Optional: show partial correlations. |
| RGen01 | EDA Summary Report (PDF/HTML) | As a user, I want a downloadable report of the full EDA workflow so I can share or archive results. | App generates a report based on selected steps (cleaning, visualizations, statistical tests). Export as PDF or HTML. |
| RGen02 | Section-based Report Customization | As a user, I want to choose which analysis sections to include in the report. | User selects checkboxes for sections (e.g. cleaning, basic EDA, advanced EDA). Subcheckboxes for the different plots to be excluded are then displayed. Report is generated accordingly. |
| RGen03 | Auto-generated Narrative | As a user, I want the report to include plain-text interpretations of the findings. | Auto-fill text summaries and/ or figure captions for plots/stats (e.g. "Feature X is normally distributed with..."). These are generated with an appropriate prompt to fetch from the OpenAI API (free key). Users can edit before export. |
| RGen04 | Save Report State | As a user, I want to save my session so I can re-download or continue editing the report later. | Save analysis steps and report content to a file (e.g. `.json` or `.pkl`). Load later to regenerate the report. |

# 

# 4\. Architecture

## 4.1 User Interface

**General Layout**

* **Header** (at the top of the application)

  * Displays the application’s title (“EDA App”).

  * Provide main controls (load, save, export).

* **Side Panel** (on the left side of the application)

  * File upload controls with drag-and-drop or directory picker.

  * Data cleaning and transformation options (handle missing values, convert data types, normalize, create new variables, select or exclude variables).

  * Filters for subsequent analyses.

  * Operations are applied to a preview of the dataset, with the option to export cleaned data at any point.

* **Main View** (central panel)

  * Displays charts, tables, statistics, and results related to the currently selected operation.

  * This view is divided into **Tabs or Subpanels** for different phases of the workflow (Exploratory Data Analysis, Hypothesis Tests, Clustering, Report Generation, etc.).

  **Exploratory Data Analysis Section** (tab within main view)

* Summary table with data types, missing values, and statistics.

* Interactive charts: histograms, boxplots, scatterplots, pie charts.

* Correlation matrix with p-values.

* Dimensionality reduction (PCA, t-SNE, UMAP) controls and visualization.

* Clustering controls (parameter settings, number of clusters).

  **Statistical Tests Section** (tab within main view)  
* Perform t-tests, ANOVA, or other significance tests.

* Display p-values, effect sizes, and diagnostics.

  **Reporting Section** (tab within main view)  
* Report builder with checkboxes to include or omit sections.

* Auto-generated text explanations alongside charts.

* Export to PDF or HTML format.

  **Session Management Section** (accessible from side panel or header)  
* Save, reload, or reset a previous analysis.

* Export cleaned data alongside reports.

## 4.2 Workflow 

**Step 1 — File Upload & Validation (Side Panel)**  
User loads a dataset (drag-and-drop or directory picker).  
The application parses and validates the format and structure.  
An initial preview is displayed in the main view.

**Step 2 — Data Cleaning & Transformation (Side Panel)**  
User handles missing values, converts data types, normalises, or adds new variables.  
All operations are previewed in real time in the main view.  
User can export cleaned data at any point.

**Step 3 — Exploratory Data Analysis (Main View — Exploratory Tab)**  
User generates charts, tables, statistics, and diagnostics to explore their data’s structure, relationships, and distributions.  
User may perform dimensionality reduction or clustering if desirable.

**Step 4 — Hypothesis Testing (Main View — Hypothesis Tests Tab)**  
User performs significance tests (t-test, ANOVA) to validate relationships in their data.  
P-values, effect sizes, and diagnostics are displayed in the main view.

**Step 5 — Report Generation (Main View — Report Tab)**  
User customises their report, choosing which sections to include.  
Auto-generated text explanations alongside charts aid interpretability.  
The application generates a downloadable PDF or HTML report.

**Step 6 — Saving or Continuing Sessions (Side Panel or Header)**  
User saves their progress and can reload their sessions for further analysis or export at a later time.  
User can export cleaned data alongside their reports for further reuse.

## 4.3 Project Structure 

1. Adopt a modular and reproducible folder structure such as:

eda\_app/  
├── core/                 \# reusable core functions go here  
│   ├── \_\_init\_\_.py       \# define standard paths (DATA\_DIR etc.)  
│   ├── upload.py         \# data upload and validation  
│   ├── cleaning.py       \# missing data, normalization, etc.  
│   ├── eda\_basic.py      \# histograms, stats, visualizations  
│   ├── eda\_advanced.py   \# PCA, clustering, etc.  
│   ├── report.py         \# PDF/HTML export functions  
│  
├── tests/                \# unit tests for core modules  
│   ├── test\_cleaning.py  
│  
├── data/                 \# local data, add to \`.gitignore\`  
├── models/               \# optional, e.g. clustering results  
├── output/               \# generated plots, logs, etc.  
│  
├── README.md  
├── requirements.txt  
├── pyproject.toml or setup.py  
├── .gitignore

2. Use `pathlib` for platform-independent paths.

\*\*Example\*\*  
In `core/__init__.py`:  
\`\`\`  
from pathlib import Path

CORE\_DIR \= Path(\_\_file\_\_).resolve().parent  
PROJECT\_DIR \= CORE\_DIR.parent  
DATA\_DIR \= PROJECT\_DIR / "data"  
MODELS\_DIR \= PROJECT\_DIR / "models"  
OUTPUT\_DIR \= PROJECT\_DIR / "output"

DATA\_DIR.mkdir(parents=True, exist\_ok=True)  
MODELS\_DIR.mkdir(parents=True, exist\_ok=True)  
OUTPUT\_DIR.mkdir(parents=True, exist\_ok=True)  
\`\`\`

3. Maintain a `tests/` directory:  
* Write unit tests for each logical module  
* Use `pytest` for test automation

\*\*Example\*\*

\`\`\`\`

def test\_normalize\_column():

    result \= normalize(\[1, 2, 3\])

    assert result.mean() \== 0

\`\`\`\`

## 4.4 Deployment

The app is planned to be deployed onto streamlit community cloud with my github linked account pascalbaertschi. Ensure that pushing to main or new PRs onto main result in an app update of the deployed state (webhook). 

