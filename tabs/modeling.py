import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import io
import base64

# Import common ML models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVR, SVC

def app():
    st.header("Modeling")
    
    # Check if data exists in session state
    if 'data' not in st.session_state:
        st.warning("Please load data in the 'Data Loading and Filtering' tab first.")
        return
    
    # Use filtered data if available, otherwise use original data
    if 'filtered_data' in st.session_state:
        df = st.session_state['filtered_data']
        st.info("Using filtered data for modeling")
    else:
        df = st.session_state['data']
        st.info("Using original data for modeling")
    
    # Task selection
    task_type = st.selectbox(
        "Select Modeling Task",
        ["Regression", "Classification"]
    )
    
    # Column selection
    all_columns = df.columns.tolist()
    
    # Select target variable
    target_col = st.selectbox("Select target variable", all_columns)
    
    # Select features
    feature_cols = st.multiselect(
        "Select feature columns",
        [col for col in all_columns if col != target_col],
        default=[col for col in all_columns if col != target_col]
    )
    
    if not feature_cols:
        st.warning("Please select at least one feature column")
        return
    
    # Get X and y
    X = df[feature_cols]
    y = df[target_col]
    
    # Identify numeric and categorical columns for preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Split data
    test_size = st.slider("Select test size ratio", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    st.write(f"Training set size: {X_train.shape[0]} samples")
    st.write(f"Test set size: {X_test.shape[0]} samples")
    
    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Model selection based on task
    if task_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Support Vector Regressor": SVR()
        }
        
        metrics = {
            "Mean Squared Error": mean_squared_error,
            "R² Score": r2_score
        }
    else:  # Classification
        models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest Classifier": RandomForestClassifier(random_state=42),
            "Decision Tree Classifier": DecisionTreeClassifier(random_state=42),
            "Support Vector Classifier": SVC(probability=True, random_state=42)
        }
        
        metrics = {
            "Accuracy": accuracy_score
        }
    
    # Model selection
    selected_model = st.selectbox("Select Model", list(models.keys()))
    
    # Create ML pipeline
    model = models[selected_model]
    
    # Hyperparameters
    st.subheader(f"Hyperparameters for {selected_model}")
    
    if selected_model in ["Random Forest Regressor", "Random Forest Classifier"]:
        n_estimators = st.slider("Number of trees", 10, 500, 100, 10)
        max_depth = st.slider("Max depth", 1, 30, 10)
        
        if task_type == "Regression":
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    
    elif selected_model in ["Decision Tree Regressor", "Decision Tree Classifier"]:
        max_depth = st.slider("Max depth", 1, 30, 10)
        min_samples_split = st.slider("Min samples split", 2, 20, 2)
        
        if task_type == "Regression":
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
        else:
            model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
    
    elif selected_model in ["Support Vector Regressor", "Support Vector Classifier"]:
        C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, 0.1)
        kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
        
        if task_type == "Regression":
            model = SVR(C=C, kernel=kernel)
        else:
            model = SVC(C=C, kernel=kernel, probability=True, random_state=42)
    
    # Create pipeline with preprocessing
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train button
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            pipeline.fit(X_train, y_train)
            
            # Save the model
            if not Path("models").exists():
                Path("models").mkdir(exist_ok=True)
                
            model_filename = f"models/{selected_model.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(pipeline, model_filename)
            st.success(f"Model successfully trained and saved to {model_filename}")
            
            # Evaluate model
            st.subheader("Model Evaluation")
            
            # Cross-validation
            st.write("Cross-validation scores (5-fold):")
            cv_scores = cross_val_score(pipeline, X, y, cv=5)
            cv_df = pd.DataFrame({
                "Fold": range(1, 6),
                "Score": cv_scores
            })
            st.write(cv_df)
            st.write(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
            
            # Test set predictions
            y_pred = pipeline.predict(X_test)
            
            # Metrics
            st.subheader("Performance Metrics")
            
            if task_type == "Regression":
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"R² Score: {r2:.4f}")
                
                # Scatter plot of actual vs predicted
                fig, ax = plt.subplots()
                plt.scatter(y_test, y_pred, alpha=0.5)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
                plt.xlabel('Actual')
                plt.ylabel('Predicted')
                plt.title('Actual vs Predicted Values')
                st.pyplot(fig)
                
                # Residual plot
                fig, ax = plt.subplots()
                residuals = y_test - y_pred
                plt.scatter(y_pred, residuals, alpha=0.5)
                plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--')
                plt.xlabel('Predicted')
                plt.ylabel('Residuals')
                plt.title('Residual Plot')
                st.pyplot(fig)
            
            else:  # Classification
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.4f}")
                
                # Classification report
                st.write("Classification Report:")
                report = classification_report(y_test, y_pred, output_dict=True)
                st.write(pd.DataFrame(report).transpose())
                
                # Confusion matrix
                st.write("Confusion Matrix:")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                plt.xlabel('Predicted Labels')
                plt.ylabel('True Labels')
                plt.title('Confusion Matrix')
                st.pyplot(fig)
                
                # ROC curve for binary classification
                if len(np.unique(y)) == 2:
                    from sklearn.metrics import roc_curve, auc
                    
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_proba)
                    roc_auc = auc(fpr, tpr)
                    
                    fig, ax = plt.subplots()
                    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC)')
                    plt.legend(loc="lower right")
                    st.pyplot(fig)
            
            # Feature importance if available
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                
                # Get feature names after preprocessing
                feature_names = []
                for name, transformer, features in preprocessor.transformers_:
                    if name == 'cat' and hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                        # For categorical features, we need to get the one-hot encoded feature names
                        cat_features = transformer.named_steps['onehot'].get_feature_names_out(features)
                        feature_names.extend(cat_features)
                    else:
                        # For numerical features, we just use the original feature names
                        feature_names.extend(features)
                
                # Simplify to use direct feature names for visualization
                importances_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': model.feature_importances_
                })
                importances_df = importances_df.sort_values('Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=importances_df, ax=ax)
                plt.title('Feature Importance')
                st.pyplot(fig)
    
    # Prediction section
    st.subheader("Make Predictions")
    
    # Check if model exists
    model_path = Path(f"models/{selected_model.lower().replace(' ', '_')}_model.pkl")
    
    if model_path.exists():
        # Load the model
        try:
            loaded_model = joblib.load(model_path)
            
            st.write("Enter values for prediction:")
            
            # Create input fields for each feature
            input_data = {}
            
            for col in feature_cols:
                if col in numeric_features:
                    # For numeric features, use a number input
                    input_data[col] = st.number_input(f"{col}", value=float(df[col].median()))
                else:
                    # For categorical features, use a selectbox
                    input_data[col] = st.selectbox(f"{col}", df[col].unique())
            
            # Create a DataFrame from input
            input_df = pd.DataFrame([input_data])
            
            if st.button("Predict"):
                # Make prediction
                prediction = loaded_model.predict(input_df)
                
                st.subheader("Prediction Result")
                st.write(f"Predicted {target_col}: {prediction[0]}")
                
                # For classification, also show probabilities if available
                if task_type == "Classification" and hasattr(loaded_model, 'predict_proba'):
                    proba = loaded_model.predict_proba(input_df)
                    st.write("Class Probabilities:")
                    
                    proba_df = pd.DataFrame(
                        proba[0],
                        index=loaded_model.classes_,
                        columns=['Probability']
                    )
                    
                    st.write(proba_df)
                    
                    # Bar chart for probabilities
                    fig, ax = plt.subplots()
                    sns.barplot(x=loaded_model.classes_, y=proba[0], ax=ax)
                    plt.xlabel('Class')
                    plt.ylabel('Probability')
                    plt.title('Prediction Probabilities')
                    plt.ylim(0, 1)
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error loading model: {e}")
    else:
        st.info("No trained model found. Please train a model first.")
