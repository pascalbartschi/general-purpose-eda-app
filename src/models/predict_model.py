
import pandas as pd
import pickle
from pathlib import Path
import json

def make_prediction(input_data, model_path=None):
    """
    Make a prediction using the specified model or the latest model
    
    Args:
        input_data: DataFrame or dict with input features
        model_path: Path to the pickled model file (optional)
    
    Returns:
        Prediction result
    """
    try:
        # If no model path is specified, try to find the latest model
        if model_path is None:
            model_dir = Path("models")
            model_files = list(model_dir.glob("*.pkl"))
            if not model_files:
                return {"error": "No model found", "prediction": None}
            
            # Sort by modification time to get the latest model
            model_path = sorted(model_files, key=lambda x: x.stat().st_mtime)[-1]
        
        # Load the model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        # Convert input data to DataFrame if it's a string or dict
        if isinstance(input_data, str):
            try:
                input_data = json.loads(input_data)
            except json.JSONDecodeError:
                return {"error": "Invalid input JSON", "prediction": None}
        
        if isinstance(input_data, dict):
            input_data = pd.DataFrame([input_data])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return {
            "prediction": prediction.tolist(),
            "model_used": str(model_path),
            "success": True
        }
        
    except Exception as e:
        return {"error": str(e), "prediction": None}