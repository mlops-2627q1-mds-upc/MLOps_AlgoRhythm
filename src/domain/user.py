import enum
from sqlalchemy import Column, Enum, Integer, String, JSON
from pydantic import BaseModel, Field, EmailStr

from src.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"

# TODO: check fields
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    stored_songs = Column(JSON, default=list, nullable=False)

    