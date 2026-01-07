from sqlalchemy import Column, String, Float, Boolean, DateTime
from database.db_manager import Base

class UserTelemetry(Base):
    __tablename__ = "user_telemetry"

    timestamp = Column(DateTime, primary_key=True)
    device_id = Column(String(30))
    posture_class = Column(Boolean, nullable=True)
    accel_magnitude = Column(Float, nullable=True)
    vision_status = Column(String(20), nullable=True)
    alert_status = Column(String(30))
