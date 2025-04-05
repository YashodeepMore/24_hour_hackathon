import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class CreditScorer:
    def __init__(self):
        self.model = GradientBoostingRegressor()
        self.preprocessor = None
        self.is_trained = False
        
    def preprocess_data(self, df):
        """Preprocess the data for model training"""
        # Separate features and target
        X = df.drop(columns=['Credit_Score'], axis=1)
        y = df['Credit_Score']
        
        # Identify numerical and categorical features
        num_features = X.select_dtypes(exclude='O').columns
        cat_features = X.select_dtypes(include="O").columns
        
        # Create transformers
        num_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, cat_features),
                ("StandardScaler", num_transformer, num_features)
            ]
        )
        
        # Transform the data
        X_processed = self.preprocessor.fit_transform(X)
        
        return X_processed, y
    
    def train(self, df):
        """Train the credit scoring model"""
        # Preprocess the data
        X, y = self.preprocess_data(df)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.is_trained = True
        
        return {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse
        }
    
    def predict(self, business_data):
        """Predict credit score for a business"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Convert input data to DataFrame
        df = pd.DataFrame([business_data])
        
        # Preprocess the data
        X = self.preprocessor.transform(df)
        
        # Make prediction
        credit_score = self.model.predict(X)[0]
        
        return max(300, min(850, credit_score))  # Ensure score is within standard range
    
    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature names from the preprocessor
        feature_names = []
        for name, transformer, features in self.preprocessor.transformers_:
            if name == "OneHotEncoder":
                # Get feature names from OneHotEncoder
                feature_names.extend(transformer.get_feature_names_out(features))
            else:
                feature_names.extend(features)
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame with feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance 