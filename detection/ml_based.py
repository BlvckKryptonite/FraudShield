"""
Machine Learning-based fraud detection algorithms
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

class MLBasedDetector:
    """Machine Learning-based fraud detection using Isolation Forest"""
    
    def __init__(self, contamination=0.1, random_state=42):
        """
        Initialize ML detector
        
        Args:
            contamination (float): Expected proportion of outliers (0.1 = 10%)
            random_state (int): Random state for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
    
    def detect_anomalies(self, df):
        """
        Detect anomalies using Isolation Forest
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            list: List of flagged transactions
        """
        try:
            # Prepare features for ML model
            features_df = self._prepare_features(df)
            
            # Train model on the data
            self._train_model(features_df)
            
            # Predict anomalies
            anomaly_scores = self.model.decision_function(features_df)
            anomaly_predictions = self.model.predict(features_df)
            
            # Create flagged transactions list
            flagged = []
            for i, (idx, row) in enumerate(df.iterrows()):
                if anomaly_predictions[i] == -1:  # Anomaly detected
                    confidence = self._calculate_confidence(anomaly_scores[i])
                    severity = self._determine_severity(anomaly_scores[i])
                    
                    flagged.append({
                        'transaction_id': row['transaction_id'],
                        'flag_type': 'ml_anomaly',
                        'reason': f'ML anomaly detected (score: {anomaly_scores[i]:.3f})',
                        'severity': severity,
                        'confidence': confidence,
                        'anomaly_score': anomaly_scores[i]
                    })
            
            return flagged
            
        except Exception as e:
            print(f"Error in ML-based detection: {str(e)}")
            return []
    
    def _prepare_features(self, df):
        """
        Prepare features for machine learning model
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Processed features
        """
        features_df = df.copy()
        
        # Convert timestamp to datetime if not already
        features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
        
        # Extract time-based features
        features_df['hour'] = features_df['timestamp'].dt.hour
        features_df['day_of_week'] = features_df['timestamp'].dt.dayofweek
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_night'] = ((features_df['hour'] >= 22) | (features_df['hour'] <= 6)).astype(int)
        
        # Amount-based features
        features_df['amount_log'] = np.log1p(features_df['amount'])
        features_df['amount_rounded'] = (features_df['amount'] % 100 == 0).astype(int)
        
        # User-based features
        user_stats = features_df.groupby('user_id')['amount'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).add_prefix('user_')
        user_stats['user_std'] = user_stats['user_std'].fillna(0)
        features_df = features_df.merge(user_stats, left_on='user_id', right_index=True)
        
        # Amount deviation from user's typical behavior
        features_df['amount_deviation'] = np.abs(
            features_df['amount'] - features_df['user_mean']
        ) / (features_df['user_std'] + 1)
        
        # Merchant-based features
        merchant_stats = features_df.groupby('merchant')['amount'].agg([
            'count', 'mean'
        ]).add_prefix('merchant_')
        features_df = features_df.merge(merchant_stats, left_on='merchant', right_index=True)
        
        # Location-based features
        location_stats = features_df.groupby('location')['amount'].agg([
            'count', 'mean'
        ]).add_prefix('location_')
        features_df = features_df.merge(location_stats, left_on='location', right_index=True)
        
        # Encode categorical variables
        categorical_columns = ['merchant', 'location']
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    features_df[col].astype(str)
                )
            else:
                # Handle unseen categories
                unique_values = set(self.label_encoders[col].classes_)
                features_df[f'{col}_encoded'] = features_df[col].apply(
                    lambda x: self.label_encoders[col].transform([x])[0] 
                    if x in unique_values else -1
                )
        
        # Select numerical features for model
        feature_columns = [
            'amount', 'amount_log', 'amount_rounded', 'amount_deviation',
            'hour', 'day_of_week', 'is_weekend', 'is_night',
            'user_count', 'user_mean', 'user_std', 'user_min', 'user_max',
            'merchant_count', 'merchant_mean',
            'location_count', 'location_mean',
            'merchant_encoded', 'location_encoded'
        ]
        
        # Ensure all feature columns exist
        available_columns = [col for col in feature_columns if col in features_df.columns]
        
        # Fill any remaining NaN values
        features_df[available_columns] = features_df[available_columns].fillna(0)
        
        self.feature_columns = available_columns
        return features_df[available_columns]
    
    def _train_model(self, features_df):
        """
        Train the Isolation Forest model
        
        Args:
            features_df (pd.DataFrame): Prepared features
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Train Isolation Forest
        self.model.fit(features_scaled)
        self.is_trained = True
    
    def _calculate_confidence(self, anomaly_score):
        """
        Calculate confidence score based on anomaly score
        
        Args:
            anomaly_score (float): Isolation Forest anomaly score
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Anomaly scores are typically between -1 and 1
        # More negative scores indicate stronger anomalies
        # Convert to confidence score (0-1 scale)
        normalized_score = max(0, min(1, (1 + anomaly_score) / 2))
        confidence = 1 - normalized_score  # Invert so lower scores = higher confidence
        return round(confidence, 3)
    
    def _determine_severity(self, anomaly_score):
        """
        Determine severity level based on anomaly score
        
        Args:
            anomaly_score (float): Isolation Forest anomaly score
            
        Returns:
            str: Severity level
        """
        if anomaly_score < -0.3:
            return 'HIGH'
        elif anomaly_score < -0.1:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def get_feature_importance(self, df):
        """
        Get feature importance analysis (simplified for Isolation Forest)
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            dict: Feature importance information
        """
        if not self.is_trained:
            features_df = self._prepare_features(df)
            self._train_model(features_df)
        
        # For Isolation Forest, we can analyze feature contribution by
        # calculating variance and correlation with anomaly scores
        features_df = self._prepare_features(df)
        features_scaled = self.scaler.transform(features_df)
        anomaly_scores = self.model.decision_function(features_scaled)
        
        feature_importance = {}
        for i, col in enumerate(self.feature_columns):
            # Calculate correlation between feature and anomaly scores
            correlation = np.corrcoef(features_scaled[:, i], anomaly_scores)[0, 1]
            feature_importance[col] = abs(correlation) if not np.isnan(correlation) else 0
        
        # Sort by importance
        sorted_importance = dict(sorted(
            feature_importance.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return sorted_importance
    
    def explain_prediction(self, transaction_row, df):
        """
        Provide explanation for a specific transaction prediction
        
        Args:
            transaction_row (pd.Series): Single transaction
            df (pd.DataFrame): Full dataset for context
            
        Returns:
            dict: Explanation of the prediction
        """
        # Prepare features for the specific transaction
        single_df = pd.DataFrame([transaction_row])
        features_df = self._prepare_features(pd.concat([df, single_df]))
        single_features = features_df.iloc[-1:][self.feature_columns]
        
        # Get prediction and score
        features_scaled = self.scaler.transform(single_features)
        anomaly_score = self.model.decision_function(features_scaled)[0]
        prediction = self.model.predict(features_scaled)[0]
        
        explanation = {
            'transaction_id': transaction_row['transaction_id'],
            'is_anomaly': prediction == -1,
            'anomaly_score': anomaly_score,
            'confidence': self._calculate_confidence(anomaly_score),
            'severity': self._determine_severity(anomaly_score),
            'contributing_factors': []
        }
        
        # Identify contributing factors (simplified approach)
        feature_values = single_features.iloc[0]
        feature_importance = self.get_feature_importance(df)
        
        # Top contributing features
        for feature, importance in list(feature_importance.items())[:5]:
            if importance > 0.1:  # Threshold for significance
                explanation['contributing_factors'].append({
                    'feature': feature,
                    'value': feature_values[feature],
                    'importance': importance
                })
        
        return explanation
