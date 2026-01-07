
import numpy as np
import pandas as pd

class FatigueComputer:
    def __init__(self):
        # Configuration based on ergonomic standards
        self.MAX_STATIC_DURATION_MINS = 60 
        self.SLUMP_THRESHOLD_NORM = 0.05
        self.MOVING_AVG_WINDOW = 20 # Frames
    
    def calculate_fatigue_score(self, history_df: pd.DataFrame) -> float:
        """
        Calculates Fatigue Index (0-100) based on time-series telemetry.
        Methodology: Hybrid Kinematic-Static Analysis.
        
        Inputs:
            history_df: DataFrame with ['timestamp', 'posture_class', 'slump_metric', 'accel_magnitude']
        """
        if history_df.empty: return 0.0
        
        # 1. Static Fatigue (Duration based)
        # Calculate how long we have been in the CURRENT posture
        last_posture = history_df.iloc[-1]['posture_class']
        # Find consecutive duration
        b = history_df['posture_class'] != history_df['posture_class'].shift()
        streak = b.cumsum()
        current_streak_data = history_df[streak == streak.iloc[-1]]
        # Ensure timestamps are compatible and timezone-aware
        # Robust conversion: handle mixed naive/aware by converting each item individually first
        timestamps_list = []
        for t in current_streak_data['timestamp']:
            ts = pd.Timestamp(t)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
            else:
                ts = ts.tz_convert('UTC')
            timestamps_list.append(ts)
            
        timestamps = pd.Series(timestamps_list)
        
        # Get start and end times
        end_time = timestamps.iloc[-1]
        start_time = timestamps.iloc[0]
        
        # Safe conversion to UTC
        if end_time.tzinfo is None:
            end_time = end_time.tz_localize('UTC')
        else:
            end_time = end_time.tz_convert('UTC')
            
        if start_time.tzinfo is None:
            start_time = start_time.tz_localize('UTC')
        else:
            start_time = start_time.tz_convert('UTC')
            
        duration_mins = (end_time - start_time).total_seconds() / 60
        
        static_score = 0
        if last_posture == "SITTING":
            # Logistic growth curve for fatigue over time 
            static_score = 100 / (1 + np.exp(-0.1 * (duration_mins - 45))) # Center at 45 mins
            
        # 2. Postural Fatigue (Slump based)
        # Rolling average of slump metric
        avg_slump = history_df['slump_metric'].rolling(window=10).mean().iloc[-1]
        
        slump_penalty = 0
        if last_posture == "SITTING" and avg_slump < self.SLUMP_THRESHOLD_NORM:
            # The closer to 0 (head down), the higher the penalty
            slump_penalty = (self.SLUMP_THRESHOLD_NORM - avg_slump) * 1000 
            
        # 3. Kinetic Fatigue (Movement variability)
        # Low variance in acceleration = Zoning out / Stiffness
        accel_variance = history_df['accel_magnitude'].tail(30).var()
        kinetic_score = 0
        if accel_variance < 0.01: # Extremely still
            kinetic_score = 20
            
        # Total Weighted Score
        final_score = (0.5 * static_score) + (0.3 * slump_penalty) + (0.2 * kinetic_score)
        return min(100.0, max(0.0, final_score))

    def get_status_label(self, score):
        if score < 30: return "FRESH"
        if score < 60: return "MODERATE_FATIGUE"
        if score < 85: return "HIGH_FATIGUE"
        return "EXHAUSTED (RISK)"
