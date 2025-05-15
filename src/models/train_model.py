import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report
import pickle
from pathlib import Path

def train_model(df, target_column, problem_type='classification', model_type='auto', test_size=0.2, random_state=42):
    """
    Train a model using the provided dataframe and target column
    
    Args:
        df: pandas DataFrame containing the data
        target_column: string name of the target column
        problem_type: 'classification' or 'regression'
        model_type: 'auto', 'random_forest', or 'linear'
        test_size: proportion of data to use for testing
        random_state: random state for reproducibility
    
    Returns:
        dict: Dictionary containing model, metrics, and other relevant info
    """
    if target_column not in df.columns:
        return {
            'error': f"Target column '{target_column}' not found in the dataframe",
            'success': False
        }
        
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle non-numeric columns
    X = pd.get_dummies(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Choose model based on problem type and model type
    model = None
    if problem_type == 'classification':
        if model_type == 'random_forest' or model_type == 'auto':
            model = RandomForestClassifier(random_state=random_state)
        elif model_type == 'linear':
            model = LogisticRegression(random_state=random_state)
    else:  # regression
        if model_type == 'random_forest' or model_type == 'auto':
            model = RandomForestRegressor(random_state=random_state)
        elif model_type == 'linear':
            model = LinearRegression()
            
    if model is None:
        return {
            'error': f"Invalid model_type '{model_type}' for problem_type '{problem_type}'",
            'success': False
        }
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {}
    if problem_type == 'classification':
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['classification_report'] = classification_report(y_test, y_pred)
    else:  # regression
        metrics['mse'] = mean_squared_error(y_test, y_pred)
        metrics['r2'] = r2_score(y_test, y_pred)
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{problem_type}_{model_type}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Return results
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': model.feature_importances_ if hasattr(model, 'feature_importances_') else None,
        'model_path': str(model_path),
        'X_columns': X.columns.tolist(),
        'success': True
    }