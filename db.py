from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///learn_assist.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

try:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
except Exception as e:
    # Streamlit Cloud may not support WAL
    print("⚠️ WAL mode not enabled:", e)


SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class QueryCache(Base):
    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True)
    query = Column(String, index=True)
    response = Column(Text)
    embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine) # Automatically create DB is it does not exist
