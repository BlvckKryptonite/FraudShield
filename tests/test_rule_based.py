"""
Unit tests for rule-based fraud detection
"""

import unittest
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection.rule_based import RuleBasedDetector

class TestRuleBasedDetector(unittest.TestCase):
    """Test cases for rule-based fraud detection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = RuleBasedDetector()
        
        # Create sample transaction data
        self.sample_data = pd.DataFrame([
            {
                'transaction_id': 'TXN001',
                'user_id': 'USER001', 
                'timestamp': '2024-07-26 09:15:00',
                'amount': 15000.00,  # High amount
                'merchant': 'Luxury Cars Inc',
                'location': 'New York'
            },
            {
                'transaction_id': 'TXN002',
                'user_id': 'USER001',
                'timestamp': '2024-07-26 09:16:00',  # 1 minute later
                'amount': 5000.00,
                'merchant': 'Electronics Store', 
                'location': 'New York'
            },
            {
                'transaction_id': 'TXN003',
                'user_id': 'USER001',
                'timestamp': '2024-07-26 09:17:00',  # 2 minutes from first
                'amount': 2500.00,
                'merchant': 'Jewelry Store',
                'location': 'New York'
            },
            {
                'transaction_id': 'TXN004',
                'user_id': 'USER002',
                'timestamp': '2024-07-26 10:00:00',
                'amount': 500.00,
                'merchant': 'Coffee Shop',
                'location': 'unknown'  # Suspicious location
            }
        ])
    
    def test_detect_high_amount_default_threshold(self):
        """Test high amount detection with default threshold"""
        flags = self.detector.detect_high_amount(self.sample_data)
        
        # Should flag TXN001 (15000 > 10000 default threshold)
        self.assertEqual(len(flags), 1)
        self.assertEqual(flags[0]['transaction_id'], 'TXN001')
        self.assertEqual(flags[0]['flag_type'], 'high_amount')
        self.assertEqual(flags[0]['severity'], 'HIGH')
    
    def test_detect_high_amount_custom_threshold(self):
        """Test high amount detection with custom threshold"""
        flags = self.detector.detect_high_amount(self.sample_data, threshold=4000)
        
        # Should flag TXN001 (15000) and TXN002 (5000)
        self.assertEqual(len(flags), 2)
        flagged_ids = [flag['transaction_id'] for flag in flags]
        self.assertIn('TXN001', flagged_ids)
        self.assertIn('TXN002', flagged_ids)
    
    def test_detect_rapid_transactions(self):
        """Test rapid transaction detection"""
        flags = self.detector.detect_rapid_transactions(self.sample_data, time_window_minutes=5)
        
        # Should flag TXN001, TXN002, TXN003 (3 transactions in 2 minutes)
        self.assertEqual(len(flags), 3)
        flagged_ids = [flag['transaction_id'] for flag in flags]
        self.assertIn('TXN001', flagged_ids)
        self.assertIn('TXN002', flagged_ids) 
        self.assertIn('TXN003', flagged_ids)
    
    def test_detect_suspicious_locations(self):
        """Test suspicious location detection"""
        flags = self.detector.detect_unexpected_locations(self.sample_data)
        
        # Should flag TXN004 for 'unknown' location
        suspicious_flags = [flag for flag in flags if flag['flag_type'] == 'suspicious_location']
        self.assertGreater(len(suspicious_flags), 0)
        
        flagged_ids = [flag['transaction_id'] for flag in suspicious_flags]
        self.assertIn('TXN004', flagged_ids)
    
    def test_no_false_positives_normal_data(self):
        """Test that normal transactions don't get flagged"""
        normal_data = pd.DataFrame([
            {
                'transaction_id': 'TXN100',
                'user_id': 'USER100',
                'timestamp': '2024-07-26 12:00:00', 
                'amount': 50.00,  # Normal amount
                'merchant': 'Coffee Shop',
                'location': 'New York'  # Normal location
            }
        ])
        
        high_amount_flags = self.detector.detect_high_amount(normal_data)
        rapid_flags = self.detector.detect_rapid_transactions(normal_data)
        location_flags = self.detector.detect_unexpected_locations(normal_data)
        
        self.assertEqual(len(high_amount_flags), 0)
        self.assertEqual(len(rapid_flags), 0)
        self.assertEqual(len(location_flags), 0)

if __name__ == '__main__':
    unittest.main()