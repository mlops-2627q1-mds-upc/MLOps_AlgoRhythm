"""Operations that coordinate user behavior and ownership checks."""

from typing import Any
from uuid import UUID

from src.domain.user import User


class UserService:
    """Coordinate user operations (the ``System`` lifeline in the UML)."""

    @staticmethod
    def add_song(
        user: User,
        song: dict[str, Any],
        authenticated_user_id: UUID | str,
    ) -> None:
        """Check that the caller owns the account, then add the song.

        ``authenticated_user_id`` must come from the application's
        authentication layer; it must not be trusted from the song request.
        """
        if str(authenticated_user_id) != user.get_id():
            raise PermissionError("A user can only add songs to their own collection.")

        user.add_song(song)
