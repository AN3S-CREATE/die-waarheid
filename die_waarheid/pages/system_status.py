"""
Status dashboard for system monitoring
"""

import streamlit as st
import time
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd

# Import health checker
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.health_check import health_checker, get_health_status, get_system_metrics


def page_system_status():
    """System status and monitoring dashboard"""
    st.header("🔍 System Status Dashboard")
    st.markdown("Real-time monitoring of system health and performance")
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # Get current health
    health = get_health_status()
    metrics = get_system_metrics()
    
    # Status overview
    st.subheader("📊 System Status Overview")
    
    # Status badges
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if health['status'] == 'healthy' else "🔴" if health['status'] == 'unhealthy' else "🟡"
        st.metric("System Status", f"{status_color} {health['status'].upper()}")
    
    with col2:
        uptime_hours = health['uptime_seconds'] / 3600
        st.metric("Uptime", f"{uptime_hours:.1f} hours")
    
    with col3:
        st.metric("CPU Usage", f"{metrics['system']['cpu_percent']:.1f}%")
    
    with col4:
        st.metric("Memory Usage", f"{metrics['system']['memory_percent']:.1f}%")
    
    # Detailed health checks
    st.subheader("🔍 Health Checks")
    
    # Disk Space
    disk = health['checks']['disk_space']
    disk_color = "green" if disk['status'] == 'healthy' else "red" if disk['status'] == 'critical' else "orange"
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Disk Space**")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = disk['usage_percent'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Disk Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': disk_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**Memory Usage**")
        memory = health['checks']['memory']
        mem_color = "green" if memory['status'] == 'healthy' else "red" if memory['status'] == 'critical' else "orange"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = memory['system_percent'],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Memory Usage (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': mem_color},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Directory Status
    st.subheader("📁 Directory Status")
    dirs = health['checks']['directories']['directories']
    
    for dir_name, dir_info in dirs.items():
        status_icon = "✅" if dir_info['status'] == 'healthy' else "❌" if dir_info['status'] == 'error' else "⚠️"
        st.write(f"{status_icon} **{dir_name.title()}**: {dir_info['path']}")
        if 'file_count' in dir_info:
            st.write(f"   Files: {dir_info['file_count']}")
        if 'error' in dir_info:
            st.error(f"   Error: {dir_info['error']}")
        if 'message' in dir_info:
            st.info(f"   {dir_info['message']}")
    
    # API Status
    api = health['checks']['api']
    st.subheader("🌐 API Connectivity")
    api_icon = "✅" if api['status'] == 'healthy' else "❌" if api['status'] == 'error' else "⚠️"
    st.write(f"{api_icon} Gemini API: {'Configured' if api.get('gemini_configured') else 'Not Configured'}")
    
    # Recent Incidents
    st.subheader("🚨 Recent Incidents")
    incidents = health_checker.get_recent_incidents(hours=24)
    
    if incidents:
        for incident in incidents[-5:]:  # Show last 5
            severity_icon = "🔴" if incident['severity'] == 'critical' else "🟡" if incident['severity'] == 'warning' else "🔵"
            st.write(f"{severity_icon} **{incident['timestamp']}** - {incident['message']}")
    else:
        st.info("No incidents in the last 24 hours")
    
    # System Metrics Table
    st.subheader("📈 Detailed Metrics")
    
    metrics_df = pd.DataFrame([
        {"Metric": "CPU Usage", "Value": f"{metrics['system']['cpu_percent']:.1f}%", "Type": "System"},
        {"Metric": "Memory Available", "Value": f"{metrics['system']['memory_available_gb']:.1f} GB", "Type": "System"},
        {"Metric": "Disk Free", "Value": f"{metrics['system']['disk_free_gb']:.1f} GB", "Type": "System"},
        {"Metric": "Process Memory", "Value": f"{metrics['process']['memory_mb']:.1f} MB", "Type": "Process"},
        {"Metric": "Process CPU", "Value": f"{metrics['process']['cpu_percent']:.1f}%", "Type": "Process"},
        {"Metric": "Thread Count", "Value": metrics['process']['threads'], "Type": "Process"},
    ])
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Refresh button
    if st.button("🔄 Refresh Now"):
        st.rerun()
