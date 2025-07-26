# FraudShield - Financial Transaction Fraud Detection Tool

## Overview

FraudShield is a Python-based command-line interface (CLI) tool designed for detecting fraudulent financial transactions. The application combines rule-based detection algorithms with machine learning techniques to identify suspicious patterns in transaction data. It processes CSV files containing transaction records and generates comprehensive reports highlighting potentially fraudulent activities.

## User Preferences

Preferred communication style: Simple, everyday language.

## Project Status

**Phase 1 - CLI Implementation**: ✅ COMPLETED (July 26, 2025)
- Full-featured Python CLI application with argument parsing
- Rule-based fraud detection (high amounts, rapid transactions, suspicious locations)
- ML-based anomaly detection using Isolation Forest algorithm
- Professional output with colored terminal displays and tabulated results
- Multiple export formats (CSV for flagged transactions, JSON for summary reports)
- Comprehensive sample dataset with realistic fraud scenarios
- All core functionalities tested and working properly

**Phase 2 - Streamlit Dashboard**: ✅ COMPLETED (July 26, 2025)
- Professional web interface with file upload and sample data options
- Interactive parameter controls (detection method, thresholds, sensitivity)
- Real-time fraud analysis with visual dashboard
- Multiple chart types (bar charts, pie charts, time series analysis)
- Geographic fraud distribution mapping with Folium
- Filterable flagged transactions table with export capabilities
- Responsive design with custom CSS styling

## System Architecture

FraudShield follows a modular architecture with clear separation of concerns:

- **CLI Interface Layer**: Main entry point (`fraudshield.py`) handles user interaction and command-line arguments
- **Detection Layer**: Contains fraud detection algorithms split into rule-based and ML-based approaches
- **Utilities Layer**: Provides data processing and report generation capabilities
- **Data Processing**: Handles CSV file loading, validation, and data cleaning
- **Report Generation**: Creates multiple output formats (CSV, JSON) for analysis results

The architecture emphasizes modularity and extensibility, allowing for easy addition of new detection algorithms or data sources.

## Key Components

### Main Application (`fraudshield.py`)
- Central orchestrator that coordinates all system components
- Provides CLI interface with colored output using colorama
- Handles user input validation and flow control
- Integrates detection algorithms and report generation

### Detection Algorithms (`detection/`)
- **Rule-Based Detector**: Implements business logic rules for fraud detection
  - High amount transaction detection
  - Rapid successive transaction detection
  - Unexpected location pattern detection
- **ML-Based Detector**: Uses Isolation Forest algorithm for anomaly detection
  - Automatic feature engineering and preprocessing
  - Unsupervised learning approach for outlier detection
  - Configurable contamination rate for sensitivity tuning

### Utilities (`utils/`)
- **Data Processor**: Handles CSV file operations and data validation
  - Multiple encoding support for file compatibility
  - Required column validation
  - Data cleaning and preprocessing
- **Report Generator**: Creates output reports in multiple formats
  - Detailed flagged transaction exports
  - Summary statistics and analytics
  - Structured JSON and CSV outputs

## Data Flow

1. **Input Processing**: CSV transaction data is loaded and validated through DataProcessor
2. **Detection Phase**: Both rule-based and ML-based detectors analyze transactions in parallel
3. **Result Aggregation**: Flagged transactions from all detectors are consolidated
4. **Report Generation**: Results are formatted and exported to various output formats
5. **User Output**: Summary statistics and flagged transactions are displayed in the CLI

The system processes data in-memory using pandas DataFrames, making it suitable for moderate-sized transaction datasets.

## External Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing support
- **scikit-learn**: Machine learning algorithms (Isolation Forest, preprocessing)
- **colorama**: Cross-platform colored terminal output
- **tabulate**: Table formatting for CLI display

### Data Processing
- **CSV format**: Primary input format for transaction data
- **JSON export**: Structured output format for integration
- **Multiple encoding support**: UTF-8, Latin-1, CP1252 for file compatibility

The application has minimal external dependencies and can run in most Python environments without additional service requirements.

## Deployment Strategy

FraudShield is designed as a standalone CLI application with the following deployment considerations:

### Local Execution
- Single Python script entry point
- Self-contained module structure
- No database or web server requirements
- Cross-platform compatibility (Windows, macOS, Linux)

### File-Based Operation
- Processes local CSV files
- Generates reports in local filesystem
- No network connectivity required for core functionality
- Reports directory automatically created for output organization

### Scalability Considerations
- In-memory processing suitable for datasets up to several GB
- Stateless design allows for easy batch processing
- Modular architecture supports future enhancements (database integration, web interface)

The current architecture prioritizes simplicity and ease of deployment while maintaining extensibility for future enterprise features.