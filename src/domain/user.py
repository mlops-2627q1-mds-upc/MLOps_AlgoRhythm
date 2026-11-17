import enum
from sqlalchemy import Column, Enum, Integer, String
from pydantic import BaseModel, Field, EmailStr


class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    USER = "USER"

# TODO: check fields
class User(BaseModel):
    user_id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    username = Column(String, unique=True, index=True)
    hahsed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.USER)
    #stored_songs: list[dict] = Field(default_factory=list, description="List of stored songs for the user")

    