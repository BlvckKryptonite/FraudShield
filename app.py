#!/usr/bin/env python3
"""
FraudShield Streamlit Dashboard
Web interface for financial transaction fraud detection
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
import io
from datetime import datetime, timedelta
import numpy as np

# Import FraudShield components
from utils.data_processor import DataProcessor
from utils.report_generator import ReportGenerator
from detection.rule_based import RuleBasedDetector
from detection.ml_based import MLBasedDetector

# Page configuration
st.set_page_config(
    page_title="FraudShield Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #1f77b4;
    }
    .alert-high { border-left-color: #dc3545; }
    .alert-medium { border-left-color: #ffc107; }
    .alert-low { border-left-color: #28a745; }
</style>
""", unsafe_allow_html=True)

class FraudShieldApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.report_generator = ReportGenerator()
        self.rule_detector = RuleBasedDetector()
        self.ml_detector = MLBasedDetector()
        
        # Initialize session state
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = None
        if 'uploaded_data' not in st.session_state:
            st.session_state.uploaded_data = None
    
    def render_header(self):
        """Render application header"""
        st.markdown("""
        <div class="main-header">
            <h1 style="color: white; margin: 0;">🛡️ FraudShield Dashboard</h1>
            <p style="color: white; margin: 0; opacity: 0.9;">Advanced Financial Transaction Fraud Detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.header("🔧 Detection Settings")
        
        # Detection method selection
        detection_method = st.sidebar.selectbox(
            "Detection Method",
            ["both", "rules", "ml"],
            help="Choose the fraud detection approach"
        )
        
        # Amount threshold
        amount_threshold = st.sidebar.number_input(
            "High Amount Threshold ($)",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Transactions above this amount will be flagged"
        )
        
        # Time window for rapid transactions
        time_window = st.sidebar.slider(
            "Rapid Transaction Window (minutes)",
            min_value=1,
            max_value=60,
            value=5,
            help="Time window to detect rapid successive transactions"
        )
        
        # ML contamination rate
        contamination_rate = st.sidebar.slider(
            "ML Anomaly Sensitivity",
            min_value=0.05,
            max_value=0.50,
            value=0.10,
            step=0.05,
            help="Expected proportion of anomalies (higher = more sensitive)"
        )
        
        return {
            'method': detection_method,
            'amount_threshold': amount_threshold,
            'time_window': time_window,
            'contamination_rate': contamination_rate
        }
    
    def render_file_upload(self):
        """Render file upload section"""
        st.header("📁 Data Upload")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Transaction CSV",
                type=['csv'],
                help="Upload a CSV file with transaction data"
            )
        
        with col2:
            if st.button("📊 Use Sample Data"):
                try:
                    sample_data = self.data_processor.load_csv('data/sample_transactions.csv')
                    st.session_state.uploaded_data = sample_data
                    st.success("Sample data loaded successfully!")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                
                # Validate data
                if self.data_processor.validate_data(df):
                    df = self.data_processor.preprocess_data(df)
                    st.session_state.uploaded_data = df
                    st.success(f"Successfully loaded {len(df)} transactions!")
                    
                    # Show data preview
                    with st.expander("📋 Data Preview"):
                        st.dataframe(df.head(), use_container_width=True)
                else:
                    st.error("Data validation failed. Please check your CSV structure.")
                    
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        return st.session_state.uploaded_data
    
    def run_analysis(self, data, settings):
        """Run fraud detection analysis"""
        if data is None:
            return None
        
        with st.spinner("🔍 Analyzing transactions for fraud patterns..."):
            try:
                flagged_transactions = []
                detection_summary = {
                    'total_transactions': len(data),
                    'detection_method': settings['method'],
                    'analysis_timestamp': datetime.now().isoformat(),
                    'parameters': settings,
                    'flags': {}
                }
                
                # Rule-based detection
                if settings['method'] in ['rules', 'both']:
                    # High amount detection
                    high_amount_flags = self.rule_detector.detect_high_amount(
                        data, settings['amount_threshold']
                    )
                    flagged_transactions.extend(high_amount_flags)
                    
                    # Rapid transactions detection
                    rapid_flags = self.rule_detector.detect_rapid_transactions(
                        data, settings['time_window']
                    )
                    flagged_transactions.extend(rapid_flags)
                    
                    # Location detection
                    location_flags = self.rule_detector.detect_unexpected_locations(data)
                    flagged_transactions.extend(location_flags)
                    
                    detection_summary['flags'].update({
                        'high_amount': len(high_amount_flags),
                        'rapid_transactions': len(rapid_flags),
                        'unexpected_location': len(location_flags)
                    })
                
                # ML-based detection
                if settings['method'] in ['ml', 'both']:
                    # Update ML detector contamination rate
                    self.ml_detector.contamination = settings['contamination_rate']
                    self.ml_detector.model.contamination = settings['contamination_rate']
                    
                    ml_flags = self.ml_detector.detect_anomalies(data)
                    flagged_transactions.extend(ml_flags)
                    
                    detection_summary['flags']['ml_anomalies'] = len(ml_flags)
                
                # Remove duplicates
                unique_flagged = self._remove_duplicate_flags(flagged_transactions)
                
                detection_summary['total_flagged'] = len(unique_flagged)
                
                return {
                    'summary': detection_summary,
                    'flagged_transactions': unique_flagged,
                    'original_data': data
                }
                
            except Exception as e:
                st.error(f"Analysis error: {str(e)}")
                return None
    
    def _remove_duplicate_flags(self, flagged_transactions):
        """Remove duplicate flagged transactions"""
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
                        break
        
        return unique_flags
    
    def render_summary_metrics(self, results):
        """Render summary metrics"""
        if not results:
            return
        
        summary = results['summary']
        total_transactions = summary['total_transactions']
        total_flagged = summary['total_flagged']
        fraud_rate = (total_flagged / total_transactions * 100) if total_transactions > 0 else 0
        
        st.header("📊 Analysis Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{total_transactions:,}",
                help="Total number of transactions analyzed"
            )
        
        with col2:
            st.metric(
                "Flagged Transactions",
                f"{total_flagged:,}",
                help="Number of potentially fraudulent transactions"
            )
        
        with col3:
            fraud_color = "normal"
            if fraud_rate > 5:
                fraud_color = "inverse"
            elif fraud_rate > 2:
                fraud_color = "off"
            
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%",
                help="Percentage of transactions flagged as potentially fraudulent"
            )
        
        with col4:
            risk_level = "LOW"
            if fraud_rate > 5:
                risk_level = "HIGH"
            elif fraud_rate > 2:
                risk_level = "MEDIUM"
            
            st.metric(
                "Risk Level",
                risk_level,
                help="Overall risk assessment based on fraud rate"
            )
    
    def render_flag_breakdown_chart(self, results):
        """Render flag breakdown visualization"""
        if not results or not results['summary']['flags']:
            return
        
        flags = results['summary']['flags']
        
        # Create flag breakdown chart
        flag_names = [name.replace('_', ' ').title() for name in flags.keys()]
        flag_counts = list(flags.values())
        
        fig = px.bar(
            x=flag_names,
            y=flag_counts,
            title="Fraud Detection Breakdown by Type",
            labels={'x': 'Flag Type', 'y': 'Count'},
            color=flag_counts,
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_severity_distribution(self, results):
        """Render severity distribution chart"""
        if not results:
            return
        
        flagged = results['flagged_transactions']
        if not flagged:
            return
        
        # Count by severity
        severity_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for flag in flagged:
            severity_counts[flag['severity']] += 1
        
        # Create pie chart
        colors = ['#dc3545', '#ffc107', '#28a745']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(severity_counts.keys()),
            values=list(severity_counts.values()),
            marker_colors=colors,
            hole=0.4
        )])
        
        fig.update_layout(
            title="Flag Severity Distribution",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_time_analysis(self, results):
        """Render time-based analysis"""
        if not results:
            return
        
        data = results['original_data']
        flagged_ids = [f['transaction_id'] for f in results['flagged_transactions']]
        
        # Add fraud flag to data
        data['is_fraud'] = data['transaction_id'].isin(flagged_ids)
        
        # Group by hour
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        hourly_stats = data.groupby('hour').agg({
            'transaction_id': 'count',
            'is_fraud': 'sum'
        }).reset_index()
        
        hourly_stats.columns = ['hour', 'total_transactions', 'fraud_transactions']
        hourly_stats['fraud_rate'] = (
            hourly_stats['fraud_transactions'] / hourly_stats['total_transactions'] * 100
        )
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Transaction Volume by Hour', 'Fraud Rate by Hour'),
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Volume chart
        fig.add_trace(
            go.Bar(
                x=hourly_stats['hour'],
                y=hourly_stats['total_transactions'],
                name='Total Transactions',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Fraud rate chart
        fig.add_trace(
            go.Scatter(
                x=hourly_stats['hour'],
                y=hourly_stats['fraud_rate'],
                mode='lines+markers',
                name='Fraud Rate %',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            title_text="Transaction Patterns by Time",
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Hour of Day", row=2, col=1)
        fig.update_yaxes(title_text="Transaction Count", row=1, col=1)
        fig.update_yaxes(title_text="Fraud Rate (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_flagged_transactions_table(self, results):
        """Render flagged transactions table"""
        if not results or not results['flagged_transactions']:
            st.info("No flagged transactions found.")
            return
        
        st.header("🚨 Flagged Transactions")
        
        # Prepare data for display
        flagged_data = []
        original_data = results['original_data']
        
        for flag in results['flagged_transactions']:
            transaction = original_data[
                original_data['transaction_id'] == flag['transaction_id']
            ].iloc[0]
            
            flagged_data.append({
                'Transaction ID': flag['transaction_id'],
                'User ID': transaction['user_id'],
                'Amount': f"${transaction['amount']:,.2f}",
                'Merchant': transaction['merchant'],
                'Location': transaction['location'],
                'Timestamp': transaction['timestamp'],
                'Flag Type': flag['flag_type'].replace('_', ' ').title(),
                'Severity': flag['severity'],
                'Confidence': f"{flag['confidence']:.2f}",
                'Reason': flag['reason']
            })
        
        df_display = pd.DataFrame(flagged_data)
        
        # Add filtering options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=['HIGH', 'MEDIUM', 'LOW'],
                default=['HIGH', 'MEDIUM', 'LOW']
            )
        
        with col2:
            flag_types = df_display['Flag Type'].unique()
            flag_filter = st.multiselect(
                "Filter by Flag Type",
                options=flag_types,
                default=flag_types
            )
        
        with col3:
            show_count = st.slider(
                "Show Top N Transactions",
                min_value=5,
                max_value=len(df_display),
                value=min(20, len(df_display))
            )
        
        # Apply filters
        filtered_df = df_display[
            (df_display['Severity'].isin(severity_filter)) &
            (df_display['Flag Type'].isin(flag_filter))
        ].head(show_count)
        
        # Display table
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "📄 Download CSV",
                data=csv_data,
                file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                "📋 Download JSON",
                data=json_data,
                file_name=f"flagged_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    def render_location_map(self, results):
        """Render location-based fraud map"""
        if not results or not results['flagged_transactions']:
            return
        
        st.header("🗺️ Fraud Geographic Distribution")
        
        # Get location data
        original_data = results['original_data']
        flagged_ids = [f['transaction_id'] for f in results['flagged_transactions']]
        
        # Sample coordinates for demo (in real implementation, you'd geocode locations)
        location_coords = {
            'New York': [40.7128, -74.0060],
            'Los Angeles': [34.0522, -118.2437],
            'Chicago': [41.8781, -87.6298],
            'Miami': [25.7617, -80.1918],
            'Seattle': [47.6062, -122.3321],
            'Dallas': [32.7767, -96.7970],
            'Boston': [42.3601, -71.0589],
            'San Francisco': [37.7749, -122.4194],
            'Las Vegas': [36.1699, -115.1398],
            'Atlanta': [33.7490, -84.3880],
            'Denver': [39.7392, -104.9903],
            'Phoenix': [33.4484, -112.0740],
            'Portland': [45.5152, -122.6784],
            'Houston': [29.7604, -95.3698],
            'Nashville': [36.1627, -86.7816],
            'Salt Lake City': [40.7608, -111.8910],
            'Albuquerque': [35.0844, -106.6504]
        }
        
        # Create map
        m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
        
        # Add markers for flagged transactions
        for flag in results['flagged_transactions']:
            transaction = original_data[
                original_data['transaction_id'] == flag['transaction_id']
            ].iloc[0]
            
            location = transaction['location']
            if location in location_coords:
                coords = location_coords[location]
                
                # Color by severity
                color = 'red' if flag['severity'] == 'HIGH' else 'orange' if flag['severity'] == 'MEDIUM' else 'yellow'
                
                folium.CircleMarker(
                    location=coords,
                    radius=8,
                    popup=f"""
                    <b>Transaction ID:</b> {flag['transaction_id']}<br>
                    <b>Amount:</b> ${transaction['amount']:,.2f}<br>
                    <b>Merchant:</b> {transaction['merchant']}<br>
                    <b>Severity:</b> {flag['severity']}<br>
                    <b>Reason:</b> {flag['reason']}
                    """,
                    color=color,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
    
    def run(self):
        """Run the main application"""
        self.render_header()
        
        # Sidebar settings
        settings = self.render_sidebar()
        
        # File upload section
        data = self.render_file_upload()
        
        if data is not None:
            # Analysis button
            if st.button("🔍 Run Fraud Analysis", type="primary"):
                results = self.run_analysis(data, settings)
                st.session_state.analysis_results = results
            
            # Display results if available
            if st.session_state.analysis_results:
                results = st.session_state.analysis_results
                
                # Summary metrics
                self.render_summary_metrics(results)
                
                # Charts and visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    self.render_flag_breakdown_chart(results)
                
                with col2:
                    self.render_severity_distribution(results)
                
                # Time analysis
                self.render_time_analysis(results)
                
                # Location map
                self.render_location_map(results)
                
                # Flagged transactions table
                self.render_flagged_transactions_table(results)
        
        else:
            st.info("👆 Please upload a CSV file or use sample data to get started.")

def main():
    """Main entry point"""
    app = FraudShieldApp()
    app.run()

if __name__ == "__main__":
    main()