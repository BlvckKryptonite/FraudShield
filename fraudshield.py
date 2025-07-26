#!/usr/bin/env python3
"""
FraudShield - Core application logic for fraud detection
Author: FraudShield Team  
Date: July 26, 2025
"""

import sys
import os
from datetime import datetime
from colorama import init, Fore, Style
from tabulate import tabulate
import pandas as pd

from utils.data_processor import DataProcessor
from utils.report_generator import ReportGenerator
from detection.rule_based import RuleBasedDetector
from detection.ml_based import MLBasedDetector

# Initialize colorama for cross-platform colored output
init(autoreset=True)

class FraudShield:
    """Main FraudShield application class"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator()
        self.rule_detector = RuleBasedDetector()
        self.ml_detector = MLBasedDetector()
        
    def print_banner(self):
        """Print application banner"""
        banner = f"""
{Fore.CYAN}
███████╗██████╗  █████╗ ██╗   ██╗██████╗ ███████╗██╗  ██╗██╗███████╗██╗     ██████╗ 
██╔════╝██╔══██╗██╔══██╗██║   ██║██╔══██╗██╔════╝██║  ██║██║██╔════╝██║     ██╔══██╗
█████╗  ██████╔╝███████║██║   ██║██║  ██║███████╗███████║██║█████╗  ██║     ██║  ██║
██╔══╝  ██╔══██╗██╔══██║██║   ██║██║  ██║╚════██║██╔══██║██║██╔══╝  ██║     ██║  ██║
██║     ██║  ██║██║  ██║╚██████╔╝██████╔╝███████║██║  ██║██║███████╗███████╗██████╔╝
╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚═╝  ╚═╝╚═╝╚══════╝╚══════╝╚═════╝ 
{Style.RESET_ALL}
{Fore.YELLOW}Financial Transaction Fraud Detection Tool v1.0{Style.RESET_ALL}
{Fore.GREEN}Protecting your financial data with advanced analytics{Style.RESET_ALL}
        """
        print(banner)
    
    def analyze_transactions(self, csv_file, method='both', amount_threshold=10000, 
                           time_window=5, export_csv=True, export_json=True):
        """
        Main analysis function
        
        Args:
            csv_file (str): Path to CSV file
            method (str): Detection method ('rules', 'ml', 'both')
            amount_threshold (float): Threshold for high-amount detection
            time_window (int): Time window in minutes for rapid transactions
            export_csv (bool): Export flagged transactions to CSV
            export_json (bool): Export summary report to JSON
        """
        try:
            # Load and validate data
            print(f"{Fore.BLUE}📊 Loading transaction data from: {csv_file}{Style.RESET_ALL}")
            df = self.data_processor.load_csv(csv_file)
            
            print(f"{Fore.GREEN}✅ Successfully loaded {len(df)} transactions{Style.RESET_ALL}")
            
            # Validate data structure
            if not self.data_processor.validate_data(df):
                print(f"{Fore.RED}❌ Data validation failed. Please check your CSV structure.{Style.RESET_ALL}")
                return False
            
            # Preprocess data
            df = self.data_processor.preprocess_data(df)
            
            flagged_transactions = []
            detection_summary = {
                'total_transactions': len(df),
                'detection_method': method,
                'analysis_timestamp': datetime.now().isoformat(),
                'parameters': {
                    'amount_threshold': amount_threshold,
                    'time_window_minutes': time_window
                },
                'flags': {}
            }
            
            # Rule-based detection
            if method in ['rules', 'both']:
                print(f"\n{Fore.BLUE}🔍 Running rule-based fraud detection...{Style.RESET_ALL}")
                
                # High amount detection
                high_amount_flags = self.rule_detector.detect_high_amount(df, amount_threshold)
                flagged_transactions.extend(high_amount_flags)
                
                # Rapid transactions detection
                rapid_flags = self.rule_detector.detect_rapid_transactions(df, time_window)
                flagged_transactions.extend(rapid_flags)
                
                # Unexpected location detection
                location_flags = self.rule_detector.detect_unexpected_locations(df)
                flagged_transactions.extend(location_flags)
                
                # Update summary
                detection_summary['flags']['high_amount'] = len(high_amount_flags)
                detection_summary['flags']['rapid_transactions'] = len(rapid_flags)
                detection_summary['flags']['unexpected_location'] = len(location_flags)
            
            # ML-based detection
            if method in ['ml', 'both']:
                print(f"\n{Fore.BLUE}🤖 Running ML-based anomaly detection...{Style.RESET_ALL}")
                
                ml_flags = self.ml_detector.detect_anomalies(df)
                flagged_transactions.extend(ml_flags)
                
                detection_summary['flags']['ml_anomalies'] = len(ml_flags)
            
            # Remove duplicates (same transaction flagged by multiple methods)
            unique_flagged = self._remove_duplicate_flags(flagged_transactions)
            
            # Display results
            self._display_results(df, unique_flagged, detection_summary)
            
            # Export reports
            if unique_flagged:
                if export_csv:
                    csv_path = self.report_generator.export_flagged_csv(unique_flagged, df)
                    print(f"\n{Fore.GREEN}💾 Flagged transactions exported to: {csv_path}{Style.RESET_ALL}")
                
                if export_json:
                    detection_summary['total_flagged'] = len(unique_flagged)
                    json_path = self.report_generator.export_summary_json(detection_summary, unique_flagged)
                    print(f"{Fore.GREEN}📋 Summary report exported to: {json_path}{Style.RESET_ALL}")
            else:
                print(f"\n{Fore.GREEN}✅ No suspicious transactions detected. No reports generated.{Style.RESET_ALL}")
            
            return True
            
        except FileNotFoundError:
            print(f"{Fore.RED}❌ Error: File '{csv_file}' not found.{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"{Fore.RED}❌ Error during analysis: {str(e)}{Style.RESET_ALL}")
            return False
    
    def _remove_duplicate_flags(self, flagged_transactions):
        """Remove duplicate flagged transactions with enhanced deduplication"""
        seen_ids = set()
        unique_flags = []
        
        for flag in flagged_transactions:
            if flag['transaction_id'] not in seen_ids:
                unique_flags.append(flag)
                seen_ids.add(flag['transaction_id'])
            else:
                # Merge reasons for duplicate flags
                for existing in unique_flags:
                    if existing['transaction_id'] == flag['transaction_id']:
                        if flag['reason'] not in existing['reason']:
                            existing['reason'] += f"; {flag['reason']}"
                        # Update severity to highest level found
                        severity_levels = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
                        if severity_levels.get(flag['severity'], 0) > severity_levels.get(existing['severity'], 0):
                            existing['severity'] = flag['severity']
                        # Update confidence to highest found
                        if flag['confidence'] > existing['confidence']:
                            existing['confidence'] = flag['confidence']
                        break
        
        return unique_flags
    
    def _generate_transaction_hash(self, transaction):
        """Generate hash for potential duplicate detection (future enhancement)"""
        import hashlib
        # Create composite key from critical fields
        key_fields = [
            str(transaction.get('user_id', '')),
            str(transaction.get('amount', '')),
            str(transaction.get('merchant', '')),
            str(transaction.get('timestamp', ''))[:16]  # Minute-level precision
        ]
        composite_key = '|'.join(key_fields)
        return hashlib.md5(composite_key.encode()).hexdigest()[:8]
    
    def _display_results(self, df, flagged_transactions, summary):
        """Display analysis results in terminal"""
        print(f"\n{Fore.CYAN}{'='*70}")
        print(f"{Fore.CYAN}                    FRAUD DETECTION RESULTS")
        print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
        
        # Summary statistics
        total_transactions = len(df)
        total_flagged = len(flagged_transactions)
        fraud_rate = (total_flagged / total_transactions * 100) if total_transactions > 0 else 0
        
        summary_data = [
            ["Total Transactions", f"{total_transactions:,}"],
            ["Flagged Transactions", f"{Fore.RED}{total_flagged:,}{Style.RESET_ALL}"],
            ["Fraud Rate", f"{Fore.YELLOW}{fraud_rate:.2f}%{Style.RESET_ALL}"],
            ["Detection Method", summary['detection_method'].upper()],
        ]
        
        print(f"\n{Fore.BLUE}📊 SUMMARY STATISTICS{Style.RESET_ALL}")
        print(tabulate(summary_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        # Flag breakdown
        if summary['flags']:
            flag_data = []
            for flag_type, count in summary['flags'].items():
                flag_name = flag_type.replace('_', ' ').title()
                color = Fore.RED if count > 0 else Fore.GREEN
                flag_data.append([flag_name, f"{color}{count:,}{Style.RESET_ALL}"])
            
            print(f"\n{Fore.BLUE}🚩 FLAG BREAKDOWN{Style.RESET_ALL}")
            print(tabulate(flag_data, headers=["Flag Type", "Count"], tablefmt="grid"))
        
        # Detailed flagged transactions
        if flagged_transactions:
            print(f"\n{Fore.RED}⚠️  FLAGGED TRANSACTIONS{Style.RESET_ALL}")
            
            # Prepare data for display
            flagged_data = []
            for flag in flagged_transactions[:10]:  # Show first 10
                transaction = df[df['transaction_id'] == flag['transaction_id']].iloc[0]
                flagged_data.append([
                    flag['transaction_id'],
                    transaction['user_id'],
                    f"${transaction['amount']:,.2f}",
                    transaction['merchant'],
                    transaction['location'],
                    flag['reason']
                ])
            
            headers = ["Transaction ID", "User ID", "Amount", "Merchant", "Location", "Reason"]
            print(tabulate(flagged_data, headers=headers, tablefmt="grid", maxcolwidths=[12, 8, 12, 15, 15, 30]))
            
            if len(flagged_transactions) > 10:
                remaining = len(flagged_transactions) - 10
                print(f"\n{Fore.YELLOW}... and {remaining} more flagged transactions{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.GREEN}✅ No suspicious transactions detected!{Style.RESET_ALL}")

def run_analysis(csv_file, method='both', amount_threshold=10000, 
                 time_window=5, export_csv=True, export_json=True, 
                 show_banner=True):
    """
    Main analysis function that can be called programmatically
    
    Args:
        csv_file (str): Path to CSV file
        method (str): Detection method ('rules', 'ml', 'both')
        amount_threshold (float): Threshold for high-amount detection
        time_window (int): Time window in minutes for rapid transactions
        export_csv (bool): Export flagged transactions to CSV
        export_json (bool): Export summary report to JSON
        show_banner (bool): Display banner output
        
    Returns:
        bool: True if analysis completed successfully
    """
    # Create FraudShield instance
    fraud_shield = FraudShield()
    
    # Print banner unless suppressed
    if show_banner:
        fraud_shield.print_banner()
    
    # Ensure reports directory exists
    os.makedirs('reports', exist_ok=True)
    
    # Run analysis
    return fraud_shield.analyze_transactions(
        csv_file=csv_file,
        method=method,
        amount_threshold=amount_threshold,
        time_window=time_window,
        export_csv=export_csv,
        export_json=export_json
    )
