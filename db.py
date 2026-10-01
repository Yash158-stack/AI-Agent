import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, LargeBinary, text
from sqlalchemy.orm import declarative_base, sessionmaker

# Always create DB inside project folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "learn_assist.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    echo=False
)

# WAL mode for concurrency
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
    document_set_id = Column(String, index=True, default="")
    prompt_version = Column(String, index=True, default="")
    model_id = Column(String, index=True, default="")
    response = Column(Text)
    embedding = Column(LargeBinary)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


Base.metadata.create_all(bind=engine)


def _ensure_cache_columns():
    required_columns = {
        "document_set_id": "ALTER TABLE query_cache ADD COLUMN document_set_id VARCHAR DEFAULT ''",
        "prompt_version": "ALTER TABLE query_cache ADD COLUMN prompt_version VARCHAR DEFAULT ''",
        "model_id": "ALTER TABLE query_cache ADD COLUMN model_id VARCHAR DEFAULT ''",
    }

    try:
        with engine.begin() as conn:
            existing = {
                row[1]
                for row in conn.execute(text("PRAGMA table_info(query_cache);"))
            }
            for name, statement in required_columns.items():
                if name not in existing:
                    conn.execute(text(statement))
    except Exception:
        pass


_ensure_cache_columns()
