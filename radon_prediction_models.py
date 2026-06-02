"""
Radon Soil Level Prediction System for Irbid, Jordan
Multi-model comparison and prediction system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from joblib import dump, load
import warnings
warnings.filterwarnings('ignore')

class RadonPredictionSystem:
    """Multi-model radon prediction system for Irbid"""
    
    def __init__(self, data_path='radon_data.csv'):
        """Initialize the prediction system"""
        self.data_path = data_path
        self.data = None
        self.models = {}
        self.predictions = {}
        self.model_performance = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load radon data"""
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully: {self.data.shape}")
            print(f"\nColumn names: {self.data.columns.tolist()}")
            print(f"\nFirst few rows:\n{self.data.head()}")
            print(f"\nData statistics:\n{self.data.describe()}")
            return self.data
        except FileNotFoundError:
            print(f"Error: {self.data_path} not found")
            return None
    
    def preprocess_data(self, target_column='radon_level'):
        """Preprocess radon data"""
        if self.data is None:
            print("No data loaded. Load data first.")
            return None
        
        # Handle missing values
        self.data.fillna(self.data.mean(), inplace=True)
        
        # Separate features and target
        X = self.data.drop(columns=[target_column, 'location_name', 'city'] 
                          if target_column in self.data.columns else [])
        
        # Keep only numeric columns
        X = X.select_dtypes(include=[np.number])
        
        if target_column not in self.data.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        
        y = self.data[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nData preprocessing complete:")
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def build_models(self, X_train, X_test, y_train, y_test):
        """Build and train multiple models"""
        models_config = {
            'Linear Regression': LinearRegression(),
            'Polynomial (degree 2)': None,  # Special handling
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
            'SVR (RBF)': SVR(kernel='rbf', C=100),
            'SVR (Linear)': SVR(kernel='linear'),
            'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        results = []
        
        for model_name, model in models_config.items():
            print(f"\nTraining {model_name}...")
            
            try:
                if model_name == 'Polynomial (degree 2)':
                    poly = PolynomialFeatures(degree=2)
                    X_train_poly = poly.fit_transform(X_train)
                    X_test_poly = poly.transform(X_test)
                    model = LinearRegression()
                    model.fit(X_train_poly, y_train)
                    y_pred = model.predict(X_test_poly)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                self.models[model_name] = model
                self.predictions[model_name] = y_pred
                self.model_performance[model_name] = {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2
                }
                
                print(f"  RMSE: {rmse:.4f}")
                print(f"  R² Score: {r2:.4f}")
                print(f"  MAE: {mae:.4f}")
                
                results.append({
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MSE': mse
                })
            
            except Exception as e:
                print(f"  Error training {model_name}: {str(e)}")
        
        return pd.DataFrame(results)
    
    def predict_for_locations(self, locations_data, target_column='radon_level'):
        """Predict radon levels for new locations"""
        predictions_dict = {'Location': locations_data.get('location_names', [])}
        
        for model_name, model in self.models.items():
            try:
                X_pred = locations_data.drop(columns=['location_names'], errors='ignore')
                X_pred = X_pred.select_dtypes(include=[np.number])
                X_pred_scaled = self.scaler.transform(X_pred)
                predictions = model.predict(X_pred_scaled)
                predictions_dict[f'{model_name}_prediction'] = predictions
            except Exception as e:
                print(f"Error predicting with {model_name}: {str(e)}")
        
        return pd.DataFrame(predictions_dict)
    
    def save_model_predictions(self, y_test, X_test_data):
        """Save predictions to CSV for each model"""
        for model_name, predictions in self.predictions.items():
            results_df = pd.DataFrame({
                'Actual': y_test.values,
                'Predicted': predictions,
                'Error': y_test.values - predictions,
                'Absolute_Error': np.abs(y_test.values - predictions),
                'Percentage_Error': np.abs((y_test.values - predictions) / y_test.values * 100)
            })
            
            filename = f'predictions_{model_name.replace(" ", "_").lower()}.csv'
            results_df.to_csv(filename, index=False)
            print(f"Saved predictions to {filename}")
    
    def compare_models(self):
        """Compare all models"""
        performance_df = pd.DataFrame(self.model_performance).T
        performance_df = performance_df.sort_values('R2', ascending=False)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        print(performance_df.to_string())
        
        return performance_df
    
    def plot_model_comparison(self, performance_df):
        """Visualize model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RMSE comparison
        performance_df['RMSE'].sort_values().plot(kind='barh', ax=axes[0, 0], color='skyblue')
        axes[0, 0].set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('RMSE')
        
        # R² comparison
        performance_df['R2'].sort_values(ascending=False).plot(kind='barh', ax=axes[0, 1], color='lightgreen')
        axes[0, 1].set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('R² Score')
        
        # MAE comparison
        performance_df['MAE'].sort_values().plot(kind='barh', ax=axes[1, 0], color='lightcoral')
        axes[1, 0].set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('MAE')
        
        # MSE comparison
        performance_df['MSE'].sort_values().plot(kind='barh', ax=axes[1, 1], color='lightyellow')
        axes[1, 1].set_title('MSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('MSE')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("Model comparison plot saved to model_comparison.png")
        plt.close()
    
    def plot_predictions_vs_actual(self):
        """Plot predictions vs actual values for each model"""
        n_models = len(self.predictions)
        fig, axes = plt.subplots(n_models, 1, figsize=(12, 4*n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, predictions) in enumerate(self.predictions.items()):
            actual = list(self.predictions.values())[0]  # Get from first model
            axes[idx].scatter(actual, predictions, alpha=0.6, s=50)
            axes[idx].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 
                          'r--', lw=2, label='Perfect Prediction')
            axes[idx].set_xlabel('Actual Radon Level')
            axes[idx].set_ylabel('Predicted Radon Level')
            axes[idx].set_title(f'{model_name} - Predictions vs Actual')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('predictions_vs_actual.png', dpi=300, bbox_inches='tight')
        print("Predictions vs actual plot saved to predictions_vs_actual.png")
        plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("RADON SOIL LEVEL PREDICTION SYSTEM - IRBID, JORDAN")
    print("="*60)
    
    # Initialize system
    system = RadonPredictionSystem(data_path='radon_data.csv')
    
    # Load and preprocess data
    data = system.load_data()
    if data is None:
        print("Please ensure radon_data.csv is in the current directory")
        return
    
    X_train, X_test, y_train, y_test, X_train_orig, X_test_orig = system.preprocess_data()
    
    # Build and train models
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    performance_df = system.build_models(X_train, X_test, y_train, y_test)
    
    # Compare models
    comparison_df = system.compare_models()
    comparison_df.to_csv('model_comparison_results.csv')
    print("\nModel comparison saved to model_comparison_results.csv")
    
    # Save predictions for each model
    print("\n" + "="*60)
    print("SAVING MODEL PREDICTIONS")
    print("="*60)
    system.save_model_predictions(y_test, X_test_orig)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    system.plot_model_comparison(comparison_df)
    system.plot_predictions_vs_actual()
    
    print("\n" + "="*60)
    print("SYSTEM READY FOR DEPLOYMENT")
    print("="*60)
    print("\nGenerated files:")
    print("  - predictions_*.csv (predictions for each model)")
    print("  - model_comparison_results.csv")
    print("  - model_comparison.png")
    print("  - predictions_vs_actual.png")


if __name__ == "__main__":
    main()
