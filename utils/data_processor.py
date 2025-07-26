"""
Data processing utilities for FraudShield
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re

class DataProcessor:
    """Data processing and validation utilities"""
    
    def __init__(self):
        self.required_columns = [
            'transaction_id', 'user_id', 'timestamp', 
            'amount', 'merchant', 'location'
        ]
    
    def load_csv(self, file_path):
        """
        Load CSV file with error handling
        
        Args:
            file_path (str): Path to CSV file
            
        Returns:
            pd.DataFrame: Loaded transaction data
        """
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError(f"Could not decode file with any of the attempted encodings: {encodings}")
            
            # Basic data cleaning
            df = df.dropna(subset=self.required_columns)
            
            return df
            
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"Error parsing CSV file: {str(e)}")
        except Exception as e:
            raise Exception(f"Unexpected error loading CSV: {str(e)}")
    
    def validate_data(self, df):
        """
        Validate data structure and content
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            bool: True if validation passes
        """
        # Check required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            print(f"Missing required columns: {missing_columns}")
            return False
        
        # Check data types and values
        validation_errors = []
        
        # Transaction ID should be unique
        if df['transaction_id'].duplicated().any():
            validation_errors.append("Duplicate transaction IDs found")
        
        # Amount should be numeric and positive
        try:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            if df['amount'].isna().any():
                validation_errors.append("Non-numeric values found in amount column")
            if (df['amount'] < 0).any():
                validation_errors.append("Negative amounts found")
        except Exception:
            validation_errors.append("Error processing amount column")
        
        # Timestamp should be parseable
        try:
            pd.to_datetime(df['timestamp'], errors='coerce')
            if pd.to_datetime(df['timestamp'], errors='coerce').isna().any():
                validation_errors.append("Invalid timestamp formats found")
        except Exception:
            validation_errors.append("Error processing timestamp column")
        
        # Check for empty required fields
        for col in self.required_columns:
            if df[col].isna().any() or (df[col] == '').any():
                validation_errors.append(f"Empty values found in {col} column")
        
        if validation_errors:
            print("Data validation errors:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        
        return True
    
    def preprocess_data(self, df):
        """
        Preprocess data for analysis
        
        Args:
            df (pd.DataFrame): Raw transaction data
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        df_processed = df.copy()
        
        # Convert timestamp to datetime
        df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
        
        # Ensure amount is numeric
        df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce')
        
        # Clean text fields
        text_columns = ['merchant', 'location']
        for col in text_columns:
            df_processed[col] = df_processed[col].astype(str).str.strip()
            df_processed[col] = df_processed[col].str.replace(r'\s+', ' ', regex=True)
        
        # Sort by timestamp for analysis
        df_processed = df_processed.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived features
        df_processed = self._add_derived_features(df_processed)
        
        return df_processed
    
    def _add_derived_features(self, df):
        """
        Add derived features for analysis
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            pd.DataFrame: Data with derived features
        """
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['timestamp'].dt.weekday.isin([5, 6])
        df['is_business_hours'] = df['hour'].between(9, 17)
        
        # Amount-based features
        df['amount_rounded'] = (df['amount'] % 100 == 0)
        df['amount_category'] = pd.cut(
            df['amount'], 
            bins=[0, 100, 500, 1000, 5000, float('inf')],
            labels=['Small', 'Medium', 'Large', 'Very Large', 'Extreme']
        )
        
        return df
    
    def get_data_summary(self, df):
        """
        Generate comprehensive data summary
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            dict: Data summary statistics
        """
        summary = {
            'total_transactions': len(df),
            'date_range': {
                'start': df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            },
            'amount_statistics': {
                'total_volume': df['amount'].sum(),
                'mean': df['amount'].mean(),
                'median': df['amount'].median(),
                'std': df['amount'].std(),
                'min': df['amount'].min(),
                'max': df['amount'].max()
            },
            'unique_counts': {
                'users': df['user_id'].nunique(),
                'merchants': df['merchant'].nunique(),
                'locations': df['location'].nunique()
            },
            'transaction_patterns': {
                'weekend_transactions': df['is_weekend'].sum(),
                'business_hours_transactions': df['is_business_hours'].sum(),
                'round_amount_transactions': df['amount_rounded'].sum()
            }
        }
        
        return summary
    
    def detect_data_quality_issues(self, df):
        """
        Detect potential data quality issues
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            dict: Data quality issues found
        """
        issues = {
            'missing_values': {},
            'outliers': {},
            'anomalies': []
        }
        
        # Missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                issues['missing_values'][col] = missing_count
        
        # Outliers in amount
        Q1 = df['amount'].quantile(0.25)
        Q3 = df['amount'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = ((df['amount'] < lower_bound) | 
                         (df['amount'] > upper_bound)).sum()
        if outliers_count > 0:
            issues['outliers']['amount'] = outliers_count
        
        # Check for suspicious patterns
        # Duplicate amounts from same user
        duplicate_amounts = df.groupby(['user_id', 'amount']).size()
        suspicious_duplicates = duplicate_amounts[duplicate_amounts >= 5]
        if len(suspicious_duplicates) > 0:
            issues['anomalies'].append(
                f"Found {len(suspicious_duplicates)} user-amount combinations with 5+ identical transactions"
            )
        
        # Check for exact timestamp duplicates
        duplicate_timestamps = df['timestamp'].duplicated().sum()
        if duplicate_timestamps > 0:
            issues['anomalies'].append(
                f"Found {duplicate_timestamps} transactions with identical timestamps"
            )
        
        return issues
    
    def export_cleaned_data(self, df, output_path):
        """
        Export cleaned and preprocessed data
        
        Args:
            df (pd.DataFrame): Cleaned transaction data
            output_path (str): Output file path
        """
        try:
            df.to_csv(output_path, index=False)
            return True
        except Exception as e:
            print(f"Error exporting cleaned data: {str(e)}")
            return False
