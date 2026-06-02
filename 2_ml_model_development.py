"""
STEP 2: MACHINE LEARNING MODEL DEVELOPMENT
This script builds and optimizes ML models for radon concentration prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class RadonPredictionModel:
    def __init__(self, data_path='radon_data_cleaned.csv'):
        """Initialize the radon prediction model"""
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.le_rock = LabelEncoder()
        self.le_soil = LabelEncoder()
        
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n" + "="*60)
        print("FEATURE PREPARATION")
        print("="*60)
        
        df_prep = self.df.copy()
        
        # Encode categorical variables
        df_prep['Rock_encoded'] = self.le_rock.fit_transform(df_prep['Rock'].fillna('Unknown'))
        df_prep['Soil_encoded'] = self.le_soil.fit_transform(df_prep['Soil'].fillna('Unknown'))
        
        # Create distance features (distance from city center)
        # Irbid center: 35.505, 32.3309
        df_prep['Distance_from_Irbid'] = np.sqrt(
            (df_prep['X'] - 35.505)**2 + (df_prep['Y'] - 32.3309)**2
        )
        
        # Feature selection
        features = ['X', 'Y', 'Elevation', 'Rock_encoded', 'Soil_encoded', 
                   'Year', 'Distance_from_Irbid']
        
        X = df_prep[features].fillna(df_prep[features].mean())
        y = df_prep['Radon'].astype(float)
        
        print(f"\nFeatures used: {features}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable shape: {y.shape}")
        
        # Feature statistics
        print("\nFeature Statistics:")
        print(X.describe())
        
        return X, y, df_prep
    
    def train_models(self, X, y):
        """Train multiple models and compare"""
        print("\n" + "="*60)
        print("MODEL TRAINING AND COMPARISON")
        print("="*60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Dictionary of models to train
        model_configs = {
            'Random Forest': {
                'model': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15],
                    'min_samples_split': [2, 5],
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                }
            },
            'AdaBoost': {
                'model': AdaBoostRegressor(n_estimators=100, random_state=42),
                'params': {
                    'n_estimators': [50, 100, 150],
                    'learning_rate': [0.01, 0.05, 0.1],
                }
            },
            'Ridge Regression': {
                'model': Ridge(alpha=1.0),
                'params': {
                    'alpha': [0.1, 1, 10, 100],
                }
            },
        }
        
        results = []
        
        for model_name, config in model_configs.items():
            print(f"\n{'='*40}")
            print(f"Training: {model_name}")
            print(f"{'='*40}")
            
            model = config['model']
            params = config['params']
            
            # Use scaled features for linear models
            if model_name in ['Ridge Regression', 'Lasso']:
                X_train_use = X_train_scaled
                X_test_use = X_test_scaled
            else:
                X_train_use = X_train
                X_test_use = X_test
            
            # GridSearchCV for hyperparameter tuning
            grid_search = GridSearchCV(
                model, params, cv=5, scoring='r2', n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train_use, y_train)
            
            best_model = grid_search.best_estimator_
            self.models[model_name] = best_model
            
            # Predictions
            y_pred_train = best_model.predict(X_train_use)
            y_pred_test = best_model.predict(X_test_use)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results.append({
                'Model': model_name,
                'Best Params': grid_search.best_params_,
                'Train R²': train_r2,
                'Test R²': test_r2,
                'Train RMSE': train_rmse,
                'Test RMSE': test_rmse,
                'Train MAE': train_mae,
                'Test MAE': test_mae,
            })
            
            print(f"Best Parameters: {grid_search.best_params_}")
            print(f"Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
            print(f"Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f}")
            print(f"Train MAE: {train_mae:.4f} | Test MAE: {test_mae:.4f}")
            
            # Cross-validation score
            cv_scores = cross_val_score(best_model, X_train_use, y_train, cv=5, scoring='r2')
            print(f"Cross-validation R² (mean ± std): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Summary comparison
        results_df = pd.DataFrame(results)
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        print(results_df.to_string(index=False))
        
        # Save results
        results_df.to_csv('model_comparison_results.csv', index=False)
        print("\n✓ Model comparison saved to: model_comparison_results.csv")
        
        # Plot comparison
        self._plot_model_comparison(results_df)
        
        return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled
    
    def _plot_model_comparison(self, results_df):
        """Plot model comparison"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # R² comparison
        axes[0].barh(results_df['Model'], results_df['Test R²'], color='skyblue', alpha=0.7)
        axes[0].set_xlabel('R² Score')
        axes[0].set_title('Test R² Comparison')
        axes[0].set_xlim([0, 1])
        
        # RMSE comparison
        axes[1].barh(results_df['Model'], results_df['Test RMSE'], color='salmon', alpha=0.7)
        axes[1].set_xlabel('RMSE (KBq/m³)')
        axes[1].set_title('Test RMSE Comparison')
        
        # MAE comparison
        axes[2].barh(results_df['Model'], results_df['Test MAE'], color='lightgreen', alpha=0.7)
        axes[2].set_xlabel('MAE (KBq/m³)')
        axes[2].set_title('Test MAE Comparison')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Model comparison plot saved: model_comparison.png")
        plt.close()
    
    def feature_importance_analysis(self, X, feature_names=None):
        """Analyze feature importance"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        if feature_names is None:
            feature_names = ['Longitude (X)', 'Latitude (Y)', 'Elevation', 
                           'Rock Type', 'Soil Type', 'Year', 'Distance from Irbid']
        
        # Use best Random Forest model
        best_model = self.models.get('Random Forest')
        if best_model is None:
            print("No Random Forest model found. Train models first.")
            return
        
        importances = best_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nFeature Importance Ranking:")
        for i, idx in enumerate(indices, 1):
            print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))
        ax.barh(range(len(indices)), importances[indices], color=colors)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title('Feature Importance for Radon Prediction (Random Forest)')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved: feature_importance.png")
        plt.close()
        
        return importances
    
    def residual_analysis(self, X_test, y_test, X_test_scaled):
        """Analyze model residuals"""
        print("\n" + "="*60)
        print("RESIDUAL ANALYSIS")
        print("="*60)
        
        best_model = self.models.get('Random Forest')
        if best_model is None:
            print("No Random Forest model found. Train models first.")
            return
        
        y_pred = best_model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Radon (KBq/m³)')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Radon (KBq/m³)')
        axes[1, 1].set_ylabel('Predicted Radon (KBq/m³)')
        axes[1, 1].set_title('Actual vs Predicted Values')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Residual analysis plot saved: residual_analysis.png")
        plt.close()
        
        # Statistics
        print(f"\nResidual Statistics:")
        print(f"Mean: {residuals.mean():.4f}")
        print(f"Std Dev: {residuals.std():.4f}")
        print(f"Min: {residuals.min():.4f}")
        print(f"Max: {residuals.max():.4f}")
        
        # Normality test
        from scipy.stats import shapiro
        stat, p_value = shapiro(residuals)
        print(f"\nShapiro-Wilk Test (Normality):")
        print(f"Test Statistic: {stat:.4f}, p-value: {p_value:.4f}")
        if p_value > 0.05:
            print("✓ Residuals appear to be normally distributed (p > 0.05)")
        else:
            print("✗ Residuals may not be normally distributed (p ≤ 0.05)")
    
    def geological_analysis(self):
        """Analyze radon levels by geological features"""
        print("\n" + "="*60)
        print("GEOLOGICAL ANALYSIS")
        print("="*60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Radon by Rock Type
        rock_stats = self.df.groupby('Rock')['Radon'].agg(['mean', 'std', 'count'])
        rock_stats = rock_stats.sort_values('mean', ascending=False)
        
        print("\nRadon by Rock Type:")
        print(rock_stats)
        
        axes[0, 0].barh(rock_stats.index, rock_stats['mean'], 
                        xerr=rock_stats['std'], capsize=5, alpha=0.7)
        axes[0, 0].set_xlabel('Average Radon (KBq/m³)')
        axes[0, 0].set_title('Average Radon Concentration by Rock Type')
        axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Radon by Soil Type
        soil_stats = self.df.groupby('Soil')['Radon'].agg(['mean', 'std', 'count'])
        soil_stats = soil_stats.sort_values('mean', ascending=False)
        
        print("\nRadon by Soil Type:")
        print(soil_stats)
        
        axes[0, 1].barh(soil_stats.index, soil_stats['mean'], 
                        xerr=soil_stats['std'], capsize=5, alpha=0.7, color='coral')
        axes[0, 1].set_xlabel('Average Radon (KBq/m³)')
        axes[0, 1].set_title('Average Radon Concentration by Soil Type')
        axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Radon by Region
        region_stats = self.df.groupby('Region')['Radon'].agg(['mean', 'std', 'count'])
        region_stats = region_stats.sort_values('mean', ascending=False)
        
        print("\nRadon by Region:")
        print(region_stats)
        
        axes[1, 0].bar(region_stats.index, region_stats['mean'], 
                       yerr=region_stats['std'], capsize=5, alpha=0.7, color='lightgreen')
        axes[1, 0].set_ylabel('Average Radon (KBq/m³)')
        axes[1, 0].set_title('Average Radon Concentration by Region')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Radon over time
        year_stats = self.df.groupby('Year')['Radon'].agg(['mean', 'count'])
        
        axes[1, 1].plot(year_stats.index, year_stats['mean'], marker='o', 
                       linewidth=2, markersize=8, color='purple')
        axes[1, 1].fill_between(year_stats.index, year_stats['mean'], alpha=0.3)
        axes[1, 1].set_xlabel('Year')
        axes[1, 1].set_ylabel('Average Radon (KBq/m³)')
        axes[1, 1].set_title('Radon Concentration Trend Over Time')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('geological_analysis.png', dpi=300, bbox_inches='tight')
        print("\n✓ Geological analysis plot saved: geological_analysis.png")
        plt.close()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n" + "="*80)
        print(" "*20 + "RADON PREDICTION MODEL - FULL ANALYSIS")
        print("="*80)
        
        # Prepare features
        X, y, df_prep = self.prepare_features()
        
        # Train models
        X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = self.train_models(X, y)
        
        # Feature importance
        feature_names = ['Longitude (X)', 'Latitude (Y)', 'Elevation', 
                        'Rock Type', 'Soil Type', 'Year', 'Distance from Irbid']
        self.feature_importance_analysis(X, feature_names)
        
        # Residual analysis
        self.residual_analysis(X_test, y_test, X_test_scaled)
        
        # Geological analysis
        self.geological_analysis()
        
        print("\n" + "="*80)
        print(" "*25 + "ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  ✓ model_comparison_results.csv")
        print("  ✓ model_comparison.png")
        print("  ✓ feature_importance.png")
        print("  ✓ residual_analysis.png")
        print("  ✓ geological_analysis.png")
        print("="*80)

if __name__ == "__main__":
    # Run analysis
    model = RadonPredictionModel('radon_data_cleaned.csv')
    model.run_full_analysis()
