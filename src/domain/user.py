"""User entity and song ownership behavior."""

from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class UserRole(str, Enum):
    """Roles available to application users."""

    USER = "user"
    ADMIN = "admin"


class User(BaseModel):
    """A user account and the songs saved by that account."""

    user_id: UUID = Field(default_factory=uuid4)
    username: str
    hashed_password: str
    role: UserRole = UserRole.USER
    stored_songs: list[dict[str, Any]] = Field(default_factory=list)

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
