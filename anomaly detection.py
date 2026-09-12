

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class MedicalAnomalyDetector:
    def __init__(self, contamination=0.05, random_state=42):
        """
        Initialize the medical anomaly detector
        
        Parameters:
        contamination: Expected proportion of high-risk patients (default 5%)
        random_state: Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.feature_importance = {}
        
    def create_sample_medical_data(self, n_patients=2000):
        """
        Create realistic sample medical data for demonstration
        """
        np.random.seed(self.random_state)
        
        # Generate normal patient data
        n_normal = int(n_patients * (1 - self.contamination))
        
        # Normal patients (healthy range)
        normal_data = {
            'age': np.random.normal(45, 15, n_normal).clip(18, 85),
            'systolic_bp': np.random.normal(120, 10, n_normal).clip(90, 140),
            'diastolic_bp': np.random.normal(80, 8, n_normal).clip(60, 90),
            'blood_glucose': np.random.normal(95, 10, n_normal).clip(70, 125),
            'total_cholesterol': np.random.normal(180, 25, n_normal).clip(120, 220),
            'hdl_cholesterol': np.random.normal(55, 10, n_normal).clip(35, 80),
            'ldl_cholesterol': np.random.normal(110, 20, n_normal).clip(70, 150),
            'bmi': np.random.normal(24, 3, n_normal).clip(18.5, 29),
            'heart_rate': np.random.normal(72, 8, n_normal).clip(60, 85),
            'triglycerides': np.random.normal(120, 30, n_normal).clip(50, 180)
        }
        
        # Add categorical variables for normal patients
        normal_data['gender'] = np.random.choice(['Male', 'Female'], n_normal, p=[0.48, 0.52])
        normal_data['smoking'] = np.random.choice(['Never', 'Former', 'Current'], 
                                                 n_normal, p=[0.6, 0.25, 0.15])
        normal_data['diabetes'] = np.random.choice(['No', 'Yes'], n_normal, p=[0.9, 0.1])
        
        # Generate high-risk patients (anomalies)
        n_high_risk = n_patients - n_normal
        
        # High-risk patients (abnormal values)
        high_risk_data = {
            'age': np.random.normal(65, 12, n_high_risk).clip(50, 90),
            'systolic_bp': np.random.choice([
                np.random.normal(160, 15, n_high_risk//2).clip(140, 200),  # Hypertension
                np.random.normal(110, 8, n_high_risk//2).clip(80, 120)    # Some normal
            ]).flatten()[:n_high_risk],
            'diastolic_bp': np.random.choice([
                np.random.normal(95, 10, n_high_risk//2).clip(90, 120),   # High
                np.random.normal(75, 5, n_high_risk//2).clip(60, 85)     # Some normal
            ]).flatten()[:n_high_risk],
            'blood_glucose': np.random.choice([
                np.random.normal(180, 30, n_high_risk//2).clip(126, 250), # Diabetes
                np.random.normal(100, 10, n_high_risk//2).clip(80, 125)   # Some normal
            ]).flatten()[:n_high_risk],
            'total_cholesterol': np.random.normal(260, 40, n_high_risk).clip(220, 350),
            'hdl_cholesterol': np.random.normal(35, 8, n_high_risk).clip(20, 45),
            'ldl_cholesterol': np.random.normal(170, 30, n_high_risk).clip(140, 250),
            'bmi': np.random.choice([
                np.random.normal(35, 5, n_high_risk//2).clip(30, 45),     # Obese
                np.random.normal(25, 3, n_high_risk//2).clip(20, 30)      # Some normal
            ]).flatten()[:n_high_risk],
            'heart_rate': np.random.choice([
                np.random.normal(95, 10, n_high_risk//2).clip(85, 120),   # High
                np.random.normal(70, 8, n_high_risk//2).clip(60, 80)      # Some normal
            ]).flatten()[:n_high_risk],
            'triglycerides': np.random.normal(220, 50, n_high_risk).clip(150, 400)
        }
        
        # Categorical variables for high-risk patients
        high_risk_data['gender'] = np.random.choice(['Male', 'Female'], n_high_risk, p=[0.6, 0.4])
        high_risk_data['smoking'] = np.random.choice(['Never', 'Former', 'Current'], 
                                                    n_high_risk, p=[0.3, 0.35, 0.35])
        high_risk_data['diabetes'] = np.random.choice(['No', 'Yes'], n_high_risk, p=[0.4, 0.6])
        
        # Combine data
        all_data = {}
        for key in normal_data.keys():
            all_data[key] = np.concatenate([normal_data[key], high_risk_data[key]])
        
        # Create labels (0 = normal, 1 = high-risk)
        all_data['true_risk_status'] = np.concatenate([
            np.zeros(n_normal), 
            np.ones(n_high_risk)
        ])
        
        # Create patient IDs
        all_data['patient_id'] = [f'P{i+1:04d}' for i in range(n_patients)]
        
        # Shuffle the data
        df = pd.DataFrame(all_data)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        return df
    
    def preprocess_data(self, df, target_col=None):
        """
        Preprocess the medical data
        """
        df_processed = df.copy()
        
        # Identify feature columns (exclude patient_id and target if present)
        exclude_cols = ['patient_id']
        if target_col and target_col in df.columns:
            exclude_cols.append(target_col)
        
        self.feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col in self.feature_columns]
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col])
            else:
                # For new data, use existing encoder
                df_processed[col] = self.label_encoders[col].transform(df_processed[col])
        
        print(f"Feature columns: {self.feature_columns}")
        print(f"Categorical columns encoded: {list(categorical_cols)}")
        
        return df_processed
    
    def fit(self, df, target_col='true_risk_status'):
        """
        Train the isolation forest model
        """
        print("=== TRAINING MEDICAL ANOMALY DETECTOR ===\n")
        
        # Preprocess data
        df_processed = self.preprocess_data(df, target_col)
        
        # Extract features
        X = df_processed[self.feature_columns]
        
        print(f"Training data shape: {X.shape}")
        print(f"Expected high-risk patients: {self.contamination:.1%}")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            max_samples='auto',
            bootstrap=False,
            n_jobs=-1,
            max_features=1.0
        )
        
        print("Training Isolation Forest model...")
        self.model.fit(X_scaled)
        
        print("Model training completed!")
        return self
    
    def predict(self, df):
        """
        Predict high-risk patients
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Preprocess data
        df_processed = self.preprocess_data(df)
        X = df_processed[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        risk_scores = self.model.score_samples(X_scaled)
        
        # Convert predictions (-1 for anomaly, 1 for normal)
        high_risk_predictions = (predictions == -1).astype(int)
        
        # Add results to original dataframe
        df_result = df.copy()
        df_result['risk_score'] = risk_scores
        df_result['predicted_high_risk'] = high_risk_predictions
        df_result['risk_level'] = df_result['predicted_high_risk'].map({
            0: 'Normal Risk', 
            1: 'High Risk'
        })
        
        return df_result
    
    def evaluate(self, df_result, target_col='true_risk_status'):
        """
        Evaluate model performance
        """
        if target_col not in df_result.columns:
            print("No ground truth available for evaluation.")
            return
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        y_true = df_result[target_col]
        y_pred = df_result['predicted_high_risk']
        
        print("\n=== MODEL EVALUATION ===")
        print(f"Total patients: {len(df_result)}")
        print(f"True high-risk patients: {y_true.sum()}")
        print(f"Predicted high-risk patients: {y_pred.sum()}")
        
        # Classification metrics
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                  target_names=['Normal Risk', 'High Risk']))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"Actual    Normal  High-Risk")
        print(f"Normal    {cm[0,0]:6d}  {cm[0,1]:9d}")
        print(f"High-Risk {cm[1,0]:6d}  {cm[1,1]:9d}")
        
        # AUC Score
        try:
            auc_score = roc_auc_score(y_true, -df_result['risk_score'])  # Negative because lower scores = higher risk
            print(f"\nAUC Score: {auc_score:.3f}")
        except:
            print("\nCould not calculate AUC score")
    
    def get_high_risk_patients(self, df_result, top_n=20):
        """
        Get the highest risk patients
        """
        high_risk_patients = df_result[df_result['predicted_high_risk'] == 1].copy()
        high_risk_patients = high_risk_patients.nsmallest(top_n, 'risk_score')
        
        print(f"\n=== TOP {top_n} HIGH-RISK PATIENTS ===")
        
        # Display relevant columns
        display_cols = ['patient_id', 'age', 'gender', 'systolic_bp', 'diastolic_bp', 
                       'blood_glucose', 'total_cholesterol', 'bmi', 'risk_score', 'risk_level']
        
        available_cols = [col for col in display_cols if col in high_risk_patients.columns]
        print(high_risk_patients[available_cols].to_string(index=False))
        
        return high_risk_patients
    
    def analyze_risk_factors(self, df_result):
        """
        Analyze which factors contribute most to high risk
        """
        print("\n=== RISK FACTOR ANALYSIS ===")
        
        # Calculate correlation between features and risk scores
        numeric_features = df_result.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features 
                       if col in self.feature_columns and col != 'risk_score']
        
        correlations = {}
        for feature in feature_cols:
            corr = abs(df_result[feature].corr(df_result['risk_score']))
            correlations[feature] = corr
        
        # Sort by correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("Feature importance (correlation with risk score):")
        for feature, corr in sorted_features[:10]:  # Top 10
            print(f"{feature:20s}: {corr:.3f}")
        
        return dict(sorted_features)
    
    def visualize_results(self, df_result):
        """
        Create comprehensive visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Medical Anomaly Detection - High Risk Patient Analysis', fontsize=16)
        
        # 1. Risk Score Distribution
        ax1 = axes[0, 0]
        normal_scores = df_result[df_result['predicted_high_risk'] == 0]['risk_score']
        high_risk_scores = df_result[df_result['predicted_high_risk'] == 1]['risk_score']
        
        ax1.hist(normal_scores, bins=50, alpha=0.7, label='Normal Risk', color='green', density=True)
        ax1.hist(high_risk_scores, bins=50, alpha=0.7, label='High Risk', color='red', density=True)
        ax1.set_xlabel('Risk Score (lower = higher risk)')
        ax1.set_ylabel('Density')
        ax1.set_title('Risk Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Age vs BMI colored by risk
        ax2 = axes[0, 1]
        if 'age' in df_result.columns and 'bmi' in df_result.columns:
            normal_mask = df_result['predicted_high_risk'] == 0
            high_risk_mask = df_result['predicted_high_risk'] == 1
            
            ax2.scatter(df_result.loc[normal_mask, 'age'], 
                       df_result.loc[normal_mask, 'bmi'],
                       c='green', alpha=0.6, label='Normal Risk', s=20)
            ax2.scatter(df_result.loc[high_risk_mask, 'age'], 
                       df_result.loc[high_risk_mask, 'bmi'],
                       c='red', alpha=0.8, label='High Risk', s=30, marker='x')
            ax2.set_xlabel('Age')
            ax2.set_ylabel('BMI')
            ax2.set_title('Age vs BMI by Risk Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Blood Pressure Analysis
        ax3 = axes[0, 2]
        if 'systolic_bp' in df_result.columns and 'diastolic_bp' in df_result.columns:
            normal_mask = df_result['predicted_high_risk'] == 0
            high_risk_mask = df_result['predicted_high_risk'] == 1
            
            ax3.scatter(df_result.loc[normal_mask, 'systolic_bp'], 
                       df_result.loc[normal_mask, 'diastolic_bp'],
                       c='green', alpha=0.6, label='Normal Risk', s=20)
            ax3.scatter(df_result.loc[high_risk_mask, 'systolic_bp'], 
                       df_result.loc[high_risk_mask, 'diastolic_bp'],
                       c='red', alpha=0.8, label='High Risk', s=30, marker='x')
            ax3.set_xlabel('Systolic BP')
            ax3.set_ylabel('Diastolic BP')
            ax3.set_title('Blood Pressure Analysis')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add healthy BP reference lines
            ax3.axhline(y=80, color='blue', linestyle='--', alpha=0.5, label='Normal DBP')
            ax3.axvline(x=120, color='blue', linestyle='--', alpha=0.5, label='Normal SBP')
        
        # 4. Cholesterol Analysis
        ax4 = axes[1, 0]
        if 'total_cholesterol' in df_result.columns:
            df_result.boxplot(column='total_cholesterol', by='risk_level', ax=ax4)
            ax4.set_title('Total Cholesterol by Risk Level')
            ax4.set_xlabel('Risk Level')
            ax4.set_ylabel('Total Cholesterol (mg/dL)')
            ax4.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='High Cholesterol Threshold')
        
        # 5. Blood Glucose Analysis
        ax5 = axes[1, 1]
        if 'blood_glucose' in df_result.columns:
            df_result.boxplot(column='blood_glucose', by='risk_level', ax=ax5)
            ax5.set_title('Blood Glucose by Risk Level')
            ax5.set_xlabel('Risk Level')
            ax5.set_ylabel('Blood Glucose (mg/dL)')
            ax5.axhline(y=126, color='red', linestyle='--', alpha=0.7, label='Diabetes Threshold')
        
        # 6. Risk Factor Importance
        ax6 = axes[1, 2]
        risk_factors = self.analyze_risk_factors(df_result)
        if risk_factors:
            factors = list(risk_factors.keys())[:8]  # Top 8 factors
            importance = [risk_factors[f] for f in factors]
            
            y_pos = np.arange(len(factors))
            ax6.barh(y_pos, importance, color='skyblue')
            ax6.set_yticks(y_pos)
            ax6.set_yticklabels(factors)
            ax6.set_xlabel('Correlation with Risk Score')
            ax6.set_title('Top Risk Factors')
            ax6.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main function demonstrating medical anomaly detection
    """
    print("=== MEDICAL ANOMALY DETECTION FOR HIGH-RISK PATIENTS ===\n")
    
    # Initialize detector
    detector = MedicalAnomalyDetector(contamination=0.05, random_state=42)
    
    # Create sample medical data
    print("1. Generating sample medical data...")
    df = detector.create_sample_medical_data(n_patients=2000)
    print(f"Generated data for {len(df)} patients")
    print(f"Features: {[col for col in df.columns if col not in ['patient_id', 'true_risk_status']]}")
    
    # Train the model
    print("\n2. Training the model...")
    detector.fit(df)
    
    # Make predictions
    print("\n3. Making predictions...")
    df_result = detector.predict(df)
    
    # Evaluate results
    print("\n4. Evaluating results...")
    detector.evaluate(df_result)
    
    # Get high-risk patients
    print("\n5. Identifying high-risk patients...")
    high_risk_patients = detector.get_high_risk_patients(df_result, top_n=15)
    
    # Analyze risk factors
    print("\n6. Analyzing risk factors...")
    risk_factors = detector.analyze_risk_factors(df_result)
    
    # Visualize results
    print("\n7. Creating visualizations...")
    detector.visualize_results(df_result)
    
    return detector, df_result

def load_your_medical_data(csv_file):
    """
    Example function to load and process your own medical data
    
    Usage:
    detector, df_result = load_your_medical_data('patient_data.csv')
    """
    print(f"Loading medical data from {csv_file}...")
    
    # Load your data
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} patient records")
    print(f"Columns: {list(df.columns)}")
    
    # Initialize detector (adjust contamination based on your expectation)
    detector = MedicalAnomalyDetector(contamination=0.05)
    
    # Train model
    detector.fit(df, target_col=None)  # No ground truth
    
    # Make predictions
    df_result = detector.predict(df)
    
    # Get high-risk patients
    high_risk_patients = detector.get_high_risk_patients(df_result)
    
    # Analyze and visualize
    detector.analyze_risk_factors(df_result)
    detector.visualize_results(df_result)
    
    return detector, df_result

if __name__ == "__main__":
    # Run the demonstration
    detector, results = main()
    
    print("\n" + "="*60)
    print("TO USE WITH YOUR OWN DATA:")
    print("="*60)
    print("""
# Example for your medical data CSV:
detector, results = load_your_medical_data('your_patient_data.csv')

# Or manually:
detector = MedicalAnomalyDetector(contamination=0.03)  # Expect 3% high-risk
df = pd.read_csv('your_data.csv')
detector.fit(df)
results = detector.predict(df)

# Get high-risk patients
high_risk = results[results['predicted_high_risk'] == 1]
print(f"Found {len(high_risk)} high-risk patients")
    """)

