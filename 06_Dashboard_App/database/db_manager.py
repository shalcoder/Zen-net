
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

Base = declarative_base()
DB_URL = "sqlite:///./guardian_system_v2.db"

class UserTelemetry(Base):
    __tablename__ = "telemetry"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    device_id = Column(String)
    
    # Raw Data
    posture_class = Column(String) # Standing, Sitting, Falling
    accel_magnitude = Column(Float) # From MPU6050
    slump_metric = Column(Float) # From MoveNet (0.0 - 1.0)
    
    # Computed Analytics
    fatigue_index = Column(Float) # 0 - 100
    risk_score = Column(Float) # 0 - 100
    vision_status = Column(String) # Normal, Fall
    alert_status = Column(String) # NORMAL, WARNING, CRITICAL

# Database Configuration
SUPABASE_URL = "postgresql://postgres:hello_world%40123@ZenNet.ai.supabase.co:5432/postgres"

try:
    # Try connecting to Supabase (Cloud)
    engine = create_engine(SUPABASE_URL, pool_pre_ping=True)
    with engine.connect() as conn:
        pass
    print("✅ Connected to Supabase Cloud Database")
except Exception as e:
    # Fallback to local SQLite if cloud is unreachable
    print(f"⚠️ Cloud DB Unreachable: {e}")
    print("📁 Falling back to local SQLite database")
    DB_URL = "sqlite:///./guardian_system_v2.db"
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
