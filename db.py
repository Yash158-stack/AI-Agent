import os
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary, text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# ✅ Always create DB inside project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "learn_assist.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# WAL only if possible
try:
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL;"))
except Exception:
    pass

SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class QueryCache(Base):
    __tablename__ = "query_cache"
    id = Column(Integer, primary_key=True)
    query = Column(String, index=True)
    response = Column(Text)
    embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)
