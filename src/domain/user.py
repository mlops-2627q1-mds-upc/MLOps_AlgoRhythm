import enum
from sqlalchemy import Column, Enum, Integer, String, JSON
from sqlalchemy.ext.mutable import MutableList
from typing import Any

from src.database import Base
from src.domain.user import UserRole


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
    stored_songs = Column(MutableList.as_mutable(JSON), default=list, nullable=False)

    def get_id(self) -> str:
        """Return the user's identifier in the string form used by the UML."""
        return str(self.user_id)

    def get_username(self) -> str:
        """Return the user's username."""
        return self.username

    def is_admin(self) -> bool:
        """Return whether this user has the administrator role."""
        return self.role == UserRole.ADMIN

    def add_song(self, song: dict[str, Any]) -> None:
        """Add a song to this user's in-memory collection.

        The calling service is responsible for checking ownership before it
        invokes this operation.
        """
        self.stored_songs.append(song)