import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Manages the database engine connection string and provides session generation dependencies

DATABSE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")

engine = create_engine(DATABSE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABSE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()