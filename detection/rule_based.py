"""
Rule-based fraud detection algorithms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

class RuleBasedDetector:
    """Rule-based fraud detection using predefined business rules"""
    
    def __init__(self):
        self.detection_rules = {
            'high_amount': self.detect_high_amount,
            'rapid_transactions': self.detect_rapid_transactions,
            'unexpected_locations': self.detect_unexpected_locations
        }
    
    def detect_high_amount(self, df, threshold=10000):
        """
        Detect transactions with unusually high amounts
        
        Args:
            df (pd.DataFrame): Transaction data
            threshold (float): Amount threshold
            
        Returns:
            list: List of flagged transactions
        """
        flagged = []
        high_amount_mask = df['amount'] > threshold
        
        for idx, row in df[high_amount_mask].iterrows():
            flagged.append({
                'transaction_id': row['transaction_id'],
                'flag_type': 'high_amount',
                'reason': f'High amount transaction: ${row["amount"]:,.2f} exceeds threshold of ${threshold:,.2f}',
                'severity': 'HIGH',
                'confidence': 0.9
            })
        
        return flagged
    
    def detect_rapid_transactions(self, df, time_window_minutes=5):
        """
        Detect rapid successive transactions from the same user
        
        Args:
            df (pd.DataFrame): Transaction data
            time_window_minutes (int): Time window in minutes
            
        Returns:
            list: List of flagged transactions
        """
        flagged = []
        
        # Sort by user and timestamp
        df_sorted = df.sort_values(['user_id', 'timestamp'])
        
        # Group by user
        for user_id, user_transactions in df_sorted.groupby('user_id'):
            if len(user_transactions) < 3:  # Need at least 3 transactions to flag
                continue
            
            # Check for rapid transactions
            timestamps = pd.to_datetime(user_transactions['timestamp'])
            
            for i in range(len(timestamps) - 2):
                # Check if 3 or more transactions occur within time window
                window_start = timestamps.iloc[i]
                window_end = window_start + timedelta(minutes=time_window_minutes)
                
                transactions_in_window = timestamps[(timestamps >= window_start) & 
                                                  (timestamps <= window_end)]
                
                if len(transactions_in_window) >= 3:
                    # Flag all transactions in this rapid sequence
                    rapid_transactions = user_transactions[
                        user_transactions['timestamp'].isin(transactions_in_window.astype(str))
                    ]
                    
                    for idx, row in rapid_transactions.iterrows():
                        flagged.append({
                            'transaction_id': row['transaction_id'],
                            'flag_type': 'rapid_transactions',
                            'reason': f'Rapid transactions: {len(transactions_in_window)} transactions in {time_window_minutes} minutes from user {user_id}',
                            'severity': 'MEDIUM',
                            'confidence': 0.8
                        })
                    break  # Avoid duplicate flags for overlapping windows
        
        return flagged
    
    def detect_unexpected_locations(self, df):
        """
        Detect transactions from unexpected or suspicious locations
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            list: List of flagged transactions
        """
        flagged = []
        
        # Define suspicious location patterns
        suspicious_patterns = [
            'unknown', 'n/a', 'null', 'anonymous', 'hidden',
            'offshore', 'foreign', 'international', 'remote'
        ]
        
        # High-risk countries/regions (simplified list)
        high_risk_locations = [
            'somalia', 'afghanistan', 'syria', 'north korea',
            'iran', 'sudan', 'libya', 'yemen'
        ]
        
        for idx, row in df.iterrows():
            location = str(row['location']).lower().strip()
            
            # Check for suspicious patterns
            if any(pattern in location for pattern in suspicious_patterns):
                flagged.append({
                    'transaction_id': row['transaction_id'],
                    'flag_type': 'suspicious_location',
                    'reason': f'Suspicious location pattern detected: {row["location"]}',
                    'severity': 'MEDIUM',
                    'confidence': 0.7
                })
                continue
            
            # Check for high-risk locations
            if any(risk_location in location for risk_location in high_risk_locations):
                flagged.append({
                    'transaction_id': row['transaction_id'],
                    'flag_type': 'high_risk_location',
                    'reason': f'High-risk location: {row["location"]}',
                    'severity': 'HIGH',
                    'confidence': 0.85
                })
                continue
        
        # Detect unusual location patterns for users
        user_location_flags = self._detect_user_location_anomalies(df)
        flagged.extend(user_location_flags)
        
        return flagged
    
    def _detect_user_location_anomalies(self, df):
        """
        Detect when users make transactions from unusual locations
        """
        flagged = []
        
        for user_id, user_transactions in df.groupby('user_id'):
            if len(user_transactions) < 2:
                continue
            
            locations = user_transactions['location'].value_counts()
            
            # If user has a dominant location (>70% of transactions)
            if len(locations) > 1 and locations.iloc[0] / len(user_transactions) > 0.7:
                dominant_location = locations.index[0]
                
                # Flag transactions from other locations
                unusual_location_transactions = user_transactions[
                    user_transactions['location'] != dominant_location
                ]
                
                for idx, row in unusual_location_transactions.iterrows():
                    flagged.append({
                        'transaction_id': row['transaction_id'],
                        'flag_type': 'unusual_user_location',
                        'reason': f'Unusual location for user {user_id}: {row["location"]} (typical: {dominant_location})',
                        'severity': 'MEDIUM',
                        'confidence': 0.75
                    })
        
        return flagged
    
    def detect_velocity_anomalies(self, df):
        """
        Detect abnormal transaction velocity patterns
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            list: List of flagged transactions
        """
        flagged = []
        
        # Calculate transaction velocities per user
        for user_id, user_transactions in df.groupby('user_id'):
            if len(user_transactions) < 2:
                continue
            
            # Sort by timestamp
            user_transactions = user_transactions.sort_values('timestamp')
            timestamps = pd.to_datetime(user_transactions['timestamp'])
            
            # Calculate time differences between consecutive transactions
            time_diffs = timestamps.diff().dt.total_seconds() / 60  # in minutes
            
            # Flag transactions with very short intervals (< 1 minute)
            for i, time_diff in enumerate(time_diffs[1:], 1):
                if time_diff < 1:  # Less than 1 minute
                    row = user_transactions.iloc[i]
                    flagged.append({
                        'transaction_id': row['transaction_id'],
                        'flag_type': 'velocity_anomaly',
                        'reason': f'Extremely rapid transaction: {time_diff:.1f} minutes after previous transaction',
                        'severity': 'HIGH',
                        'confidence': 0.9
                    })
        
        return flagged
    
    def detect_round_number_bias(self, df):
        """
        Detect suspicious round number transactions (potential money laundering)
        
        Args:
            df (pd.DataFrame): Transaction data
            
        Returns:
            list: List of flagged transactions
        """
        flagged = []
        
        # Define round number thresholds
        round_amounts = [100, 500, 1000, 5000, 10000, 50000, 100000]
        
        for idx, row in df.iterrows():
            amount = row['amount']
            
            # Check if amount is exactly a round number above certain threshold
            if amount >= 1000 and amount in round_amounts:
                flagged.append({
                    'transaction_id': row['transaction_id'],
                    'flag_type': 'round_number_bias',
                    'reason': f'Suspicious round amount: ${amount:,.2f} (potential structuring)',
                    'severity': 'LOW',
                    'confidence': 0.6
                })
        
        return flagged
