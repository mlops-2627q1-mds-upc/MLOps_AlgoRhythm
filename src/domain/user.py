from enum import Enum
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, EmailStr


class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"

class User(BaseModel):
    user_id: UUID = Field(default_factory=uuid4, description="Unique identifier for the user")
    username: str = Field(..., description="Username of the user")
    role: UserRole = Field(default=UserRole.USER, description="Role of the user, either 'admin' or 'user'")
    stored_songs: list[dict] = Field(default_factory=list, description="List of stored songs for the user")


def get_current_user() -> User:
    """
    Simula la obtención del usuario actual.
    En un escenario real, esto podría implicar la decodificación de un token JWT o la consulta a una base de datos.
    """
    # Para propósitos de prueba, devolvemos un usuario admin por defecto
    return User(username="test_admin", role=UserRole.ADMIN)
    