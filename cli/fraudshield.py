#!/usr/bin/env python3
"""
FraudShield CLI - Command-line interface for fraud detection
Author: FraudShield Team
Date: July 26, 2025
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fraudshield import run_analysis

def main():
    """CLI entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description="FraudShield - Financial Transaction Fraud Detection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/fraudshield.py data/sample_transactions.csv
  python cli/fraudshield.py data/transactions.csv --method rules --threshold 5000
  python cli/fraudshield.py data/transactions.csv --method ml --no-csv
  python cli/fraudshield.py data/transactions.csv --time-window 10 --no-json
        """
    )
    
    parser.add_argument(
        'csv_file',
        help='Path to the CSV file containing transaction data'
    )
    
    parser.add_argument(
        '--method', '-m',
        choices=['rules', 'ml', 'both'],
        default='both',
        help='Detection method to use (default: both)'
    )
    
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=10000,
        help='Amount threshold for high-value transaction detection (default: 10000)'
    )
    
    parser.add_argument(
        '--time-window', '-w',
        type=int,
        default=5,
        help='Time window in minutes for rapid transaction detection (default: 5)'
    )
    
    parser.add_argument(
        '--no-csv',
        action='store_true',
        help='Skip CSV export of flagged transactions'
    )
    
    parser.add_argument(
        '--no-json',
        action='store_true',
        help='Skip JSON export of summary report'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress banner output'
    )
    
    args = parser.parse_args()
    
    # Run analysis using core logic
    success = run_analysis(
        csv_file=args.csv_file,
        method=args.method,
        amount_threshold=args.threshold,
        time_window=args.time_window,
        export_csv=not args.no_csv,
        export_json=not args.no_json,
        show_banner=not args.quiet
    )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()