

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os

# FIX IMPORT PATHS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy.orm import Session
from database.db_manager import SessionLocal, UserTelemetry, init_db
from streamlit_autorefresh import st_autorefresh

# Initialize database tables
init_db()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Guardian AI | Medical Monitor",
    page_icon=":material/health_and_safety:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- MODERN UI STYLING (Glassmorphism + Animations) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap" rel="stylesheet">
<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">

<style>
    /* GLOBAL THEME */
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* REMOVE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* GLASSMORPHISM CARDS */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* METRICS STYLING */
    .metric-value {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 14px;
        color: #A0A0A0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }
    
    /* STATUS INDICATORS */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 50px;
        font-size: 14px;
        font-weight: 600;
    }
    .status-safe { background: rgba(0, 255, 127, 0.15); color: #00FF7F; border: 1px solid #00FF7F33; }
    .status-warn { background: rgba(255, 165, 0, 0.15); color: #FFA500; border: 1px solid #FFA50033; }
    .status-danger { 
        background: rgba(255, 69, 58, 0.15); color: #FF453A; 
        border: 1px solid #FF453A33; 
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 69, 58, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 69, 58, 0); }
    }
    
    /* ICONS */
    i { margin-right: 8px; }
    
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def get_status_badge(text):
    icon = "fa-check-circle"
    style = "status-safe"
    
    if "RISK" in text or "FALL" in text:
        icon = "fa-exclamation-triangle"
        style = "status-danger"
    elif "FATIGUE" in text or "WARNING" in text:
        icon = "fa-clock"
        style = "status-warn"
        
    return f"""
    <div class='glass-card'>
        <div class='metric-label'>User Status</div>
        <div class='{style} status-badge'>
            <i class="fas {icon}"></i> {text}
        </div>
    </div>
    """

def get_metric_card(label, value, icon_class, suffix=""):
    return f"""
    <div class='glass-card'>
        <div class='metric-label'><i class="{icon_class}"></i> {label}</div>
        <div class='metric-value'>{value}<span style='font-size: 20px; color: #666;'>{suffix}</span></div>
    </div>
    """

# --- DATA IO ---
def get_data():
    db = SessionLocal()
    data = db.query(UserTelemetry).order_by(UserTelemetry.timestamp.desc()).limit(200).all()
    db.close()
    if not data: return pd.DataFrame()
    return pd.DataFrame([vars(d) for d in data]).drop(columns=['_sa_instance_state'])

def generate_ai_insight(df):
    if df.empty: return "Initializing Guardian Intelligence...", "normal"
    
    latest = df.iloc[0]
    avg_fatigue = df['fatigue_index'].mean()
    
    # 1. Posture Trend
    sitting_ratio = (df['posture_class'] == "SITTING").mean()
    
    if latest['risk_score'] > 80:
        return "CRITICAL: Imminent Fall Risk detected. Verify user stability immediately.", "danger"
    
    if sitting_ratio > 0.8:
        return "Insight: User is highly sedentary (80%+). Recommend 2-minute mobility break to prevent vascular stiffness.", "warn"
        
    if avg_fatigue > 60:
        return "Care Note: Increasing fatigue detected over last 15 mins. Check hydration levels and environment temperature.", "warn"
        
    return "Protocol Status: User exhibiting stable mobility patterns. No intervention required.", "safe"

def get_trend_indicator(df):
    if len(df) < 10: return "fas fa-minus", "#666", "Stable"
    current = df['risk_score'].iloc[:5].mean()
    previous = df['risk_score'].iloc[5:15].mean()
    
    if current > previous + 5:
        return "fas fa-arrow-up", "#FF453A", "Rising"
    elif current < previous - 5:
        return "fas fa-arrow-down", "#00FF7F", "Declining"
    return "fas fa-minus", "#666", "Stable"

# --- MAIN APP LAYOUT ---

# Top Navigation Bar with Final Decision Box
st.markdown("""
<div style='display: flex; justify-content: space-between; align-items: center; padding: 20px 0; border-bottom: 1px solid #333; margin-bottom: 30px;'>
    <div style='display: flex; flex-direction: column;'>
        <div style='font-size: 24px; font-weight: 800; background: linear-gradient(90deg, #FFF 0%, #CCC 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            <i class="fas fa-shield-virus" style="color: #4facfe;"></i> GUARDIAN AI
        </div>
        <div style='font-size: 14px; color: #888; margin-top: 4px;'>Privacy-Aware Multi-Modal Edge-AI Elderly Safety System | A ZenNet Product</div>
    </div>
    <div style='display: flex; gap: 15px; align-items: center;'>
        <div style='color: #666; font-size: 12px;'>
            <i class="fas fa-circle" style="color: #00FF7F; font-size: 8px; margin-right: 5px;"></i> SYSTEM ONLINE
        </div>
        <div id='final-decision-box' style='padding: 12px 24px; border-radius: 12px; font-size: 15px; font-weight: 800; background: rgba(0,255,127,0.15); color: #00FF7F; border: 2px solid #00FF7F; min-width: 200px; text-align: center;'>
            <div style='font-size: 10px; opacity: 0.7; margin-bottom: 2px;'>FINAL DECISION</div>
            <div><i class="fas fa-shield-alt"></i> INITIALIZING...</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### <i class='fas fa-cog'></i> Settings", unsafe_allow_html=True)
    st.toggle("Real-Time Stream", value=True)
    st.divider()
    st.markdown("""
    <div style='font-size: 12px; color: #666;'>
        Connected Devices:<br>
        <span style='color: #fff;'>• RPI-Guardian-01</span><br>
        <span style='color: #fff;'>• ESP32-Sensor-A</span>
    </div>
    """, unsafe_allow_html=True)

# Main Refresh Loop
st_autorefresh(interval=1000, key="data_refresh")

df = get_data()

if df.empty:
    st.info("Waiting for telemetry stream...")
else:
    # --- LOGIC SEPARATION: Multi-Modal Stream Handling ---
    # We must treat Vision (RPi) and Wearable (ESP32) as separate independent streams
    
    # 1. Filter streams
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    current_time = pd.Timestamp.now(tz='UTC')
    
    # Get latest from Webcam Node
    cam_df = df[df['device_id'].str.contains("RPI", case=False, na=False)]
    if not cam_df.empty:
        latest_cam = cam_df.iloc[0]
        time_diff = (current_time - latest_cam['timestamp']).total_seconds()
        cam_active = time_diff < 10  # Active if msg in last 10s
    else:
        latest_cam = None
        cam_active = False

    # Get latest from Wearable Node
    wear_df = df[df['device_id'].str.contains("ESP32", case=False, na=False)]
    if not wear_df.empty:
        latest_wear = wear_df.iloc[0]
        time_diff = (current_time - latest_wear['timestamp']).total_seconds()
        wear_active = time_diff < 10
    else:
        latest_wear = None
        wear_active = False

    # --- FUSION ENGINE (Dashboard Side) ---
    is_vision_fall = False
    is_imu_impact = False
    
    # Check Vision (Camera)
    if latest_cam is not None:
        is_vision_fall = (latest_cam['vision_status'] == "Fall") or ("FALLING" in str(latest_cam['posture_class']))
    
    # Check IMU (Wearable) - Check BOTH posture AND impact
    if latest_wear is not None:
        # Fall is verified if EITHER posture is FALLING OR high impact detected
        wearable_fall = ("FALLING" in str(latest_wear['posture_class']))
        high_impact = (latest_wear['accel_magnitude'] > 2.2)
        is_imu_impact = wearable_fall or high_impact
        
    # Debug: Print fusion status (you can remove this later)
    if is_vision_fall or is_imu_impact:
        st.sidebar.write(f"🔍 DEBUG:")
        st.sidebar.write(f"Vision Fall: {is_vision_fall}")
        st.sidebar.write(f"IMU Impact: {is_imu_impact}")
        if latest_cam is not None:
            st.sidebar.write(f"Cam: {latest_cam['vision_status']}, {latest_cam['posture_class']}")
        if latest_wear is not None:
            st.sidebar.write(f"Wear: {latest_wear['posture_class']}, {latest_wear['accel_magnitude']:.2f}G")
        
    # --- UI DISPLAY VARIABLES ---
    # If a device is offline, we fallback to the "latest" available record for Global KPIs
    # But for Fusion Verification, we show "OFFLINE"
    
    latest_global = df.iloc[0] # For generic KPIs like Fatigue / Mobility Index
    
    # --- TABS LAYOUT ---
    tab1, tab2, tab3 = st.tabs([":material/monitor: Live Monitor", ":material/analytics: Analytics", ":material/list: System Logs"])
    
    # TAB 1: LIVE MONITOR (KPIs)
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        
        with k1:
            # Live Activity now shows ESP32 (Wearable) posture with confidence
            if latest_wear is not None:
                val = latest_wear['posture_class']
                conf = latest_wear['risk_score']  # Using risk_score as confidence proxy
                conf_bar = f"<div style='margin-top: 5px; font-size: 11px; color: #888;'>Confidence: {conf:.1f}%</div><div style='width: 100%; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top: 3px;'><div style='width: {min(conf, 100)}%; height: 100%; background: linear-gradient(90deg, #4facfe, #00f2fe); border-radius: 2px;'></div></div>"
                st.markdown(get_metric_card("Live Activity", val, "fas fa-walking") + conf_bar, unsafe_allow_html=True)
            else:
                st.markdown(get_metric_card("Live Activity", "WAITING...", "fas fa-walking"), unsafe_allow_html=True)
                
        with k2:
            # Vision Status now shows Camera vision with confidence
            if latest_cam is not None:
                val = latest_cam['vision_status']
                conf = latest_cam['risk_score']  # Using risk_score as confidence
                conf_bar = f"<div style='margin-top: 5px; font-size: 11px; color: #888;'>Confidence: {conf:.1f}%</div><div style='width: 100%; height: 3px; background: rgba(255,255,255,0.1); border-radius: 2px; margin-top: 3px;'><div style='width: {min(conf, 100)}%; height: 100%; background: linear-gradient(90deg, #4facfe, #00f2fe); border-radius: 2px;'></div></div>"
                st.markdown(get_metric_card("Vision Status", val, "fas fa-eye") + conf_bar, unsafe_allow_html=True)
            else:
                st.markdown(get_metric_card("Vision Status", "NO SIGNAL", "fas fa-eye"), unsafe_allow_html=True)
            
        with k3:
            trend_icon, trend_color, trend_label = get_trend_indicator(df)
            trend_html = f"<div style='font-size: 11px; color: {trend_color}; margin-top: -10px;'><i class='{trend_icon}'></i> Trend: {trend_label}</div>"
            st.markdown(get_metric_card("Impact/Risk", f"{latest_global['risk_score']:.1f}", "fas fa-exclamation-circle") + trend_html, unsafe_allow_html=True)
            
        with k4:
            mobility_index = round(100 - (latest_global['fatigue_index'] * 0.5), 1)
            st.markdown(get_metric_card("Mobility Index", f"{mobility_index}", "fas fa-heartbeat", suffix="%"), unsafe_allow_html=True)
            
        # AI Insight Banner
        insight_text, insight_style = generate_ai_insight(df)
        
        st.markdown(f"""
        <div class='glass-card' style='border-left: 5px solid {("#FF453A" if insight_style=="danger" else "#FFA500" if insight_style=="warn" else "#00FF7F")}; background: rgba(255,255,255,0.03);'>
            <div class='metric-label'><i class="fas fa-brain" style="color: #4facfe;"></i> Proactive Care Suggestion</div>
            <div style='font-size: 16px; font-weight: 500; color: #EEE; margin-top: 8px;'>{insight_text}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # --- SENSOR FUSION VERIFICATION CENTER ---
        st.markdown("<div class='metric-label'><i class='fas fa-microchip'></i> Sensor Fusion Verification Center</div>", unsafe_allow_html=True)
        f1, f2 = st.columns(2)
        
        with f1:
            v_color = "#FF453A" if is_vision_fall else "#00FF7F"
            v_msg = "FALL DETECTED" if is_vision_fall else "STABLE MONITORING"
            if not cam_active:
                v_color = "#666"
                v_msg = "DEVICE OFFLINE"
                
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid {v_color}33;'>
                <div style='font-size: 12px; color: #888;'>WEBCAM (RPI)</div>
                <div style='font-size: 18px; color: {v_color}; font-weight: 600;'>
                    <i class='fas fa-video'></i> {v_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with f2:
            i_color = "#FF453A" if is_imu_impact else "#00FF7F"
            i_msg = "IMU IMPACT HIT" if is_imu_impact else "NORMAL KINEMATICS"
            if not wear_active:
                i_color = "#666"
                i_msg = "DEVICE OFFLINE"
                
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid {i_color}33;'>
                <div style='font-size: 12px; color: #888;'>WEARABLE (ESP32)</div>
                <div style='font-size: 18px; color: {i_color}; font-weight: 600;'>
                    <i class='fas fa-user-clock'></i> {i_msg}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Fusion Verification Logic
        # EMERGENCY if: Camera sees fall (with or without wearable confirmation)
        verified_fall = is_vision_fall  # Vision is primary indicator
        
        # If both sensors agree on fall, it's highest confidence
        both_sensors_fall = is_vision_fall and is_imu_impact
        
        # Anomaly if only wearable detects (possible device drop, not actual fall)
        anomaly = is_imu_impact and not is_vision_fall
        
        status_color = "#FF453A" if verified_fall else "#FFA500" if anomaly else "#00FF7F"
        
        if verified_fall:
            if both_sensors_fall:
                status_msg = "VERIFIED FALL (DUAL SENSOR)"
            else:
                status_msg = "VISION FALL DETECTED (EMERGENCY)"
        elif anomaly:
            status_msg = "WEARABLE SPIKE (CHECKING...)"
        else:
            status_msg = "DUAL-SENSOR SYNC: OK"
            
        if not cam_active and not wear_active:
            status_msg = "SYSTEM OFFLINE"
            status_color = "#333"
        
        # Update the top-right Final Decision box using Streamlit components
        final_status_text = "🚨 EMERGENCY" if verified_fall else "⚠️ CHECKING" if anomaly else "✅ ALL CLEAR"
        final_status_bg = f"rgba(255,69,58,0.2)" if verified_fall else "rgba(255,165,0,0.2)" if anomaly else "rgba(0,255,127,0.15)"
        final_status_border = f"#FF453A" if verified_fall else "#FFA500" if anomaly else "#00FF7F"
        
        # Use HTML component to force update
        import streamlit.components.v1 as components
        components.html(f"""
        <script>
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                finalStatus: '{final_status_text}',
                bg: '{final_status_bg}',
                color: '{final_status_border}'
            }}, '*');
            
            // Direct DOM update
            setTimeout(() => {{
                const badge = window.parent.document.getElementById('final-decision-box');
                if (badge) {{
                    badge.innerHTML = '<div style="font-size: 10px; opacity: 0.7; margin-bottom: 2px;">FINAL DECISION</div><div><i class="fas fa-shield-alt"></i> {final_status_text}</div>';
                    badge.style.background = '{final_status_bg}';
                    badge.style.color = '{final_status_border}';
                    badge.style.borderColor = '{final_status_border}';
                }}
            }}, 100);
        </script>
        """, height=0)
        
        st.markdown(f"""
        <div style='text-align: center; padding: 10px; margin-top: 15px; border-radius: 8px; background: {status_color}11; color: {status_color}; font-weight: 800; border: 1px dashed {status_color}55;'>
            <i class='fas fa-shield-alt'></i> {status_msg}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info(":material/info: **Live Analysis**: System is monitoring real-time telemetry from connected Edge devices.")

    # TAB 2: ANALYTICS (Charts)
    with tab2:
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.markdown("<div class='metric-label'><i class='fas fa-chart-area'></i> Real-Time Fatigue Analytics</div>", unsafe_allow_html=True)
            
            fig = px.area(df, x='timestamp', y='fatigue_index', 
                          height=350,
                          color_discrete_sequence=['#4facfe'])
            
            fig.add_hline(y=80, line_dash="dash", line_color="#ff4b4b", 
                          annotation_text="Critical Risk", annotation_position="top right")
            fig.update_yaxes(range=[0, 105]) # Fixed range for clarity

            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': '#A0A0A0'},
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(":material/help: **Guide**: This graph shows the worker's Fatigue Score (0-100) over time. "
                       "Scores **above 80** (Red Line) indicate exhaustion and high fall risk.")
            
        with c2:
            st.markdown("<div class='metric-label'><i class='fas fa-chart-pie'></i> Activity Distribution</div>", unsafe_allow_html=True)
            
            # Donut Chart
            posture_counts = df['posture_class'].value_counts().reset_index()
            posture_counts.columns = ['class', 'count']
            
            fig_pie = px.pie(posture_counts, values='count', names='class', hole=0.7,
                             color_discrete_sequence=['#00f2fe', '#4facfe', '#00c6ff'])
            
            fig_pie.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=20, b=50),
                font={'color': '#FFF'}
            )
            st.plotly_chart(fig_pie, width="stretch")

    # TAB 3: LOGS
    with tab3:
        st.markdown("<div class='metric-label' style='margin-left: 10px; margin-top: 20px;'><i class='fas fa-list'></i> Recent Telemetry Logs</div>", unsafe_allow_html=True)
        
        # Custom Styled Table
        table_df = df[['timestamp', 'device_id', 'posture_class', 'alert_status', 'fatigue_index']].head(10)
        
        st.markdown("""
        <style>
            .dataframe { font-family: 'Inter'; width: 100%; border-collapse: collapse; }
            .dataframe td, .dataframe th { padding: 12px; text-align: left; border-bottom: 1px solid #333; color: #DDD; }
            .dataframe th { background-color: #1E1E1E; color: #888; font-weight: 600; text-transform: uppercase; font-size: 12px; }
            .dataframe tr:hover { background-color: #262626; }
        </style>
        """, unsafe_allow_html=True)
        
        st.dataframe(table_df, width="stretch", hide_index=True)
