#!/bin/bash
# Script to set up Conda environment for EDA App

echo "Setting up Conda environment for EDA App..."

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed or not available in PATH."
    echo "Please install Miniconda or Anaconda and try again."
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

if [ $? -ne 0 ]; then
    echo "Failed to create conda environment."
    echo "Trying alternative method..."
    
    # Try creating environment manually
    conda create -n eda_app python=3.9 -y
    if [ $? -ne 0 ]; then
        echo "Failed to create conda environment."
        exit 1
    fi
    
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate eda_app
    if [ $? -ne 0 ]; then
        echo "Failed to activate conda environment."
        exit 1
    fi
    
    # Install requirements
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install requirements."
        exit 1
    fi
else
    # Activate environment
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate eda_app
fi

echo ""
echo "Environment set up successfully!"
echo "To activate the environment, run: conda activate eda_app"
echo "To run the app, run: streamlit run app.py"
echo ""
