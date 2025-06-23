# General Purpose EDA App

This is a comprehensive, production-ready Exploratory Data Analysis (EDA) application built with Streamlit. The app enables users to upload, clean, analyze, and generate reports for datasets with an intuitive interface.
**This has to be cleaned up**
Kaggle links: 
* https://www.kaggle.com/datasets/nelgiriyewithana/billionaires-statistics-dataset
* https://www.kaggle.com/c/titanic/
* https://www.kaggle.com/datasets/firedmosquito831/student-academic-performance-simulation-4000

## Features

- **Flexible Data Upload**: Support for .csv, .xlsx, and .xls formats up to 5GB
- **Data Cleaning & Transformation**: 
  - Non-standard missing value detection
  - Data type conversion
  - Normalization
  - Feature creation
  - Column selection
- **Basic EDA**:
  - Descriptive statistics
  - Distribution visualization
  - Outlier detection
  - Feature relationship analysis
  - Time series analysis
- **Advanced EDA**:
  - Correlation matrix with p-values
  - Dimensionality reduction (PCA, t-SNE, UMAP)
  - Clustering (KMeans, DBSCAN)
  - Hypothesis testing (t-tests, ANOVA)
- **Report Generation**:
  - Customizable HTML reports
  - AI-generated explanations (with OpenAI API)
  - Session saving and loading

## Project Structure

```
general_purpose_eda_app/
├── core/                 # Core functionality modules
│   ├── __init__.py       # Defines standard paths
│   ├── upload.py         # Data upload functionality
│   ├── cleaning.py       # Data cleaning functionality
│   ├── eda_basic.py      # Basic EDA functionality
│   ├── eda_advanced.py   # Advanced EDA functionality
│   ├── report.py         # Report generation functionality
│
├── tests/                # Unit tests
│   ├── test_cleaning.py  # Tests for cleaning module
│
├── data/                 # Local data directory
├── models/               # Models directory
├── output/               # Generated outputs directory
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Dependencies
├── environment.yml       # Conda environment specification
├── README.md             # Project documentation
```

## Environment Management

This project uses Python 3.9 and includes both `requirements.txt` and `environment.yml` files for dependency management:

- **requirements.txt**: For pip-based installation
- **environment.yml**: For conda-based installation (recommended)

Using a virtual environment is strongly recommended to avoid conflicts with other projects. The conda environment approach provides better isolation and is more robust for packages with complex dependencies like WeasyPrint.

### Troubleshooting Package Installation

Some packages may require additional system dependencies:

#### Note on PDF Report Generation

PDF report generation has been temporarily deprecated due to system dependency issues with GTK libraries on Windows. Currently, the app only generates HTML reports, which provide the same content and can be printed to PDF using your browser's print function.

**Future PDF support:**
- PDF report generation will be reintroduced in future versions using alternative approaches that don't require complex system dependencies
- See the "Future Features" section at the end of this README for more details

**Current workaround for PDF:**
1. Generate and download the HTML report
2. Open the HTML report in your web browser
3. Use your browser's Print function (Ctrl+P)
4. Select "Save as PDF" as the printer option

This provides equivalent functionality while we work on a more robust PDF generation solution.

#### Other Dependencies

- **UMAP**: May require a C++ compiler on some systems.
- **Matplotlib**: May require additional system libraries for rendering.

If you encounter issues with these packages, you can try installing them separately after setting up the basic environment.

### Automated Setup Scripts

The project includes automated setup scripts to simplify environment creation:

- **Windows**: Run `setup_env.bat`
- **Linux/Mac**: Run `bash setup_env.sh`

These scripts will:
1. Check if Conda is installed
2. Create the `eda_app` environment using `environment.yml`
3. Fall back to manual creation + pip install if the environment.yml approach fails
4. Activate the environment

## Getting Started

### Prerequisites

- Python 3.8+
- Conda or Pip package manager

### Installation

#### Option 1: Using pip

1. Clone the repository
   ```
   git clone https://github.com/pascalbaertschi/general_purpose_eda_app.git
   cd general_purpose_eda_app
   ```

2. Install dependencies
   ```
   pip install -r requirements.txt
   ```

3. Run the application
   ```
   streamlit run app.py
   ```

#### Option 2: Using Conda (recommended)

1. Clone the repository
   ```
   git clone https://github.com/pascalbaertschi/general_purpose_eda_app.git
   cd general_purpose_eda_app
   ```

2. Create a new Conda environment using the environment.yml file
   ```
   conda env create -f environment.yml
   conda activate eda_app
   ```

   Alternatively, you can create the environment manually:
   ```
   conda create -n eda_app python=3.9
   conda activate eda_app
   pip install -r requirements.txt
   ```

3. Run the application
   ```
   streamlit run app.py
   ```

## Usage

1. **Data Upload**: Use the upload interface to select a CSV or Excel file
2. **Data Cleaning**: Clean and transform your data
3. **Basic EDA**: Explore distributions, relationships, and statistics
4. **Advanced EDA**: Perform more complex analyses
5. **Report Generation**: Create and download customized reports

## Development Principles

- **Reusability**: Modular structure with single-responsibility functions
- **Readability**: Clean, PEP8-compliant code with type hints
- **Reproducibility**: Controlled randomness with fixed seeds
- **Performance**: Vectorized operations for efficient data processing
- **Data-Agnostic**: Works with diverse datasets using intelligent inference

## Deployment

The app is deployed to Streamlit Community Cloud and automatically updates when changes are pushed to the main branch.

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## Future Features

The following features are planned for future releases:

### PDF Report Generation

PDF report generation is temporarily deprecated due to system dependency issues with GTK libraries on Windows. Future versions will reintroduce PDF export using one of these approaches:

1. Improved GTK integration with better cross-platform support
2. Alternative PDF generation libraries that don't require GTK (e.g., pdfkit with wkhtmltopdf)
3. Browser-based PDF generation

### Other Planned Features

- Enhanced AI-driven insights using more advanced models
- Integration with cloud storage providers
- More advanced statistical modeling capabilities
- Interactive dashboards with Streamlit components

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Data processing with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Visualization with [Plotly](https://plotly.com/python/), [Matplotlib](https://matplotlib.org/), and [Seaborn](https://seaborn.pydata.org/)
- Machine learning with [Scikit-learn](https://scikit-learn.org/)
