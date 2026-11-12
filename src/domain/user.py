from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    user_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the user")
    username: str
    email: 
    