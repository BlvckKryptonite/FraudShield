"""
Report generation utilities for FraudShield
"""

import pandas as pd
import json
import os
from datetime import datetime

class ReportGenerator:
    """Generate various reports for fraud detection results"""
    
    def __init__(self, reports_dir='reports'):
        self.reports_dir = reports_dir
        # Ensure reports directory exists
        os.makedirs(reports_dir, exist_ok=True)
    
    def export_flagged_csv(self, flagged_transactions, original_df):
        """
        Export flagged transactions to CSV
        
        Args:
            flagged_transactions (list): List of flagged transaction details
            original_df (pd.DataFrame): Original transaction data
            
        Returns:
            str: Path to exported CSV file
        """
        # Create detailed flagged transactions dataframe
        flagged_details = []
        
        for flag in flagged_transactions:
            # Get original transaction data
            transaction = original_df[
                original_df['transaction_id'] == flag['transaction_id']
            ].iloc[0]
            
            # Combine original data with flag information
            flagged_detail = {
                'transaction_id': transaction['transaction_id'],
                'user_id': transaction['user_id'],
                'timestamp': transaction['timestamp'],
                'amount': transaction['amount'],
                'merchant': transaction['merchant'],
                'location': transaction['location'],
                'flag_type': flag['flag_type'],
                'reason': flag['reason'],
                'severity': flag['severity'],
                'confidence': flag['confidence']
            }
            
            # Add anomaly score if available (ML detection)
            if 'anomaly_score' in flag:
                flagged_detail['anomaly_score'] = flag['anomaly_score']
            
            flagged_details.append(flagged_detail)
        
        # Create DataFrame and sort by severity and confidence
        flagged_df = pd.DataFrame(flagged_details)
        
        # Sort by severity (HIGH -> MEDIUM -> LOW) and confidence (descending)
        severity_order = {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        flagged_df['severity_numeric'] = flagged_df['severity'].map(severity_order)
        flagged_df = flagged_df.sort_values(
            ['severity_numeric', 'confidence'], 
            ascending=[False, False]
        ).drop('severity_numeric', axis=1)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'flagged_transactions_{timestamp}.csv'
        filepath = os.path.join(self.reports_dir, filename)
        
        # Export to CSV
        flagged_df.to_csv(filepath, index=False)
        
        return filepath
    
    def export_summary_json(self, detection_summary, flagged_transactions):
        """
        Export summary report to JSON
        
        Args:
            detection_summary (dict): Detection summary statistics
            flagged_transactions (list): List of flagged transactions
            
        Returns:
            str: Path to exported JSON file
        """
        # Enhance summary with detailed statistics
        enhanced_summary = detection_summary.copy()
        
        # Add flagged transaction details
        enhanced_summary['flagged_transactions'] = []
        
        # Aggregate statistics by flag type
        flag_type_stats = {}
        severity_stats = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for flag in flagged_transactions:
            # Add to detailed list (first 50 for JSON size management)
            if len(enhanced_summary['flagged_transactions']) < 50:
                enhanced_summary['flagged_transactions'].append({
                    'transaction_id': flag['transaction_id'],
                    'flag_type': flag['flag_type'],
                    'reason': flag['reason'],
                    'severity': flag['severity'],
                    'confidence': flag['confidence']
                })
            
            # Update statistics
            flag_type = flag['flag_type']
            if flag_type not in flag_type_stats:
                flag_type_stats[flag_type] = {'count': 0, 'avg_confidence': 0}
            
            flag_type_stats[flag_type]['count'] += 1
            severity_stats[flag['severity']] += 1
        
        # Calculate average confidence by flag type
        for flag_type, stats in flag_type_stats.items():
            confidences = [f['confidence'] for f in flagged_transactions 
                          if f['flag_type'] == flag_type]
            stats['avg_confidence'] = sum(confidences) / len(confidences) if confidences else 0
            stats['avg_confidence'] = round(stats['avg_confidence'], 3)
        
        enhanced_summary['flag_type_statistics'] = flag_type_stats
        enhanced_summary['severity_distribution'] = severity_stats
        
        # Add risk assessment
        total_flagged = len(flagged_transactions)
        total_transactions = enhanced_summary['total_transactions']
        fraud_rate = (total_flagged / total_transactions * 100) if total_transactions > 0 else 0
        
        if fraud_rate > 5:
            risk_level = 'HIGH'
        elif fraud_rate > 2:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        enhanced_summary['risk_assessment'] = {
            'fraud_rate_percentage': round(fraud_rate, 2),
            'risk_level': risk_level,
            'high_severity_count': severity_stats['HIGH'],
            'recommendations': self._generate_recommendations(
                fraud_rate, flag_type_stats, severity_stats
            )
        }
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'fraud_detection_summary_{timestamp}.json'
        filepath = os.path.join(self.reports_dir, filename)
        
        # Export to JSON
        with open(filepath, 'w') as f:
            json.dump(enhanced_summary, f, indent=2, default=str)
        
        return filepath
    
    def _generate_recommendations(self, fraud_rate, flag_type_stats, severity_stats):
        """
        Generate recommendations based on analysis results
        
        Args:
            fraud_rate (float): Percentage of flagged transactions
            flag_type_stats (dict): Statistics by flag type
            severity_stats (dict): Statistics by severity
            
        Returns:
            list: List of recommendations
        """
        recommendations = []
        
        # General recommendations based on fraud rate
        if fraud_rate > 5:
            recommendations.append(
                "HIGH ALERT: Fraud rate exceeds 5%. Immediate investigation recommended."
            )
            recommendations.append(
                "Consider implementing additional transaction monitoring controls."
            )
        elif fraud_rate > 2:
            recommendations.append(
                "Elevated fraud rate detected. Increase monitoring frequency."
            )
        
        # Specific recommendations based on flag types
        if 'high_amount' in flag_type_stats and flag_type_stats['high_amount']['count'] > 0:
            recommendations.append(
                "Review high-amount transaction thresholds and approval processes."
            )
        
        if 'rapid_transactions' in flag_type_stats and flag_type_stats['rapid_transactions']['count'] > 0:
            recommendations.append(
                "Implement velocity controls to prevent rapid successive transactions."
            )
        
        if any('location' in flag_type for flag_type in flag_type_stats.keys()):
            recommendations.append(
                "Consider implementing geo-location verification for unusual location patterns."
            )
        
        if 'ml_anomaly' in flag_type_stats and flag_type_stats['ml_anomaly']['count'] > 0:
            recommendations.append(
                "ML model detected behavioral anomalies. Consider manual review of flagged patterns."
            )
        
        # Severity-based recommendations
        if severity_stats['HIGH'] > 0:
            recommendations.append(
                f"Priority investigation required for {severity_stats['HIGH']} high-severity alerts."
            )
        
        if not recommendations:
            recommendations.append(
                "No significant fraud patterns detected. Continue regular monitoring."
            )
        
        return recommendations
    
    def generate_executive_summary(self, detection_summary, flagged_transactions):
        """
        Generate executive summary report
        
        Args:
            detection_summary (dict): Detection summary
            flagged_transactions (list): Flagged transactions
            
        Returns:
            str: Executive summary text
        """
        total_transactions = detection_summary['total_transactions']
        total_flagged = len(flagged_transactions)
        fraud_rate = (total_flagged / total_transactions * 100) if total_transactions > 0 else 0
        
        # Count by severity
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for flag in flagged_transactions:
            severity_counts[flag['severity']] += 1
        
        summary = f"""
FRAUD DETECTION EXECUTIVE SUMMARY
=================================
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Detection Method: {detection_summary['detection_method'].upper()}

KEY METRICS:
- Total Transactions Analyzed: {total_transactions:,}
- Flagged Transactions: {total_flagged:,}
- Fraud Rate: {fraud_rate:.2f}%

SEVERITY BREAKDOWN:
- High Risk: {severity_counts['HIGH']:,} transactions
- Medium Risk: {severity_counts['MEDIUM']:,} transactions  
- Low Risk: {severity_counts['LOW']:,} transactions

RISK ASSESSMENT:
"""
        
        if fraud_rate > 5:
            summary += "🔴 HIGH RISK - Immediate action required\n"
        elif fraud_rate > 2:
            summary += "🟡 MEDIUM RISK - Increased monitoring recommended\n"
        else:
            summary += "🟢 LOW RISK - Normal monitoring sufficient\n"
        
        # Add top flag types
        flag_types = {}
        for flag in flagged_transactions:
            flag_type = flag['flag_type']
            flag_types[flag_type] = flag_types.get(flag_type, 0) + 1
        
        if flag_types:
            summary += "\nTOP FRAUD INDICATORS:\n"
            sorted_flags = sorted(flag_types.items(), key=lambda x: x[1], reverse=True)
            for flag_type, count in sorted_flags[:3]:
                summary += f"- {flag_type.replace('_', ' ').title()}: {count} cases\n"
        
        return summary
    
    def export_detailed_report(self, detection_summary, flagged_transactions, original_df):
        """
        Export comprehensive detailed report
        
        Args:
            detection_summary (dict): Detection summary
            flagged_transactions (list): Flagged transactions  
            original_df (pd.DataFrame): Original data
            
        Returns:
            str: Path to detailed report file
        """
        # Generate executive summary
        exec_summary = self.generate_executive_summary(detection_summary, flagged_transactions)
        
        # Detailed analysis by flag type
        flag_analysis = {}
        for flag in flagged_transactions:
            flag_type = flag['flag_type']
            if flag_type not in flag_analysis:
                flag_analysis[flag_type] = []
            flag_analysis[flag_type].append(flag)
        
        # Create detailed report content
        report_content = exec_summary + "\n\n"
        
        report_content += "DETAILED ANALYSIS BY FLAG TYPE:\n"
        report_content += "=" * 50 + "\n\n"
        
        for flag_type, flags in flag_analysis.items():
            report_content += f"{flag_type.replace('_', ' ').upper()}:\n"
            report_content += f"Count: {len(flags)}\n"
            
            # Sample transactions
            report_content += "Sample flagged transactions:\n"
            for flag in flags[:5]:  # Show first 5
                transaction = original_df[
                    original_df['transaction_id'] == flag['transaction_id']
                ].iloc[0]
                
                report_content += f"- ID: {flag['transaction_id']}, "
                report_content += f"Amount: ${transaction['amount']:,.2f}, "
                report_content += f"User: {transaction['user_id']}, "
                report_content += f"Reason: {flag['reason']}\n"
            
            if len(flags) > 5:
                report_content += f"... and {len(flags) - 5} more\n"
            
            report_content += "\n"
        
        # Generate filename and save
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'detailed_fraud_report_{timestamp}.txt'
        filepath = os.path.join(self.reports_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(report_content)
        
        return filepath
