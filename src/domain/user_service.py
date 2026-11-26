"""Operations that coordinate user behavior and ownership checks."""

from typing import Any
from uuid import UUID

from src.domain.user import User, UserRole


class UserService:
    """Coordinate user operations (the ``System`` lifeline in the UML)."""

    @staticmethod
    def ensure_owner(target_user: User, actor_id: UUID | str) -> None:
        """Raise a PermissionError if the actor is not the owner of the target user."""
        if str(actor_id) != target_user.get_id():
            raise PermissionError("A user can only access or modify their own account.")
        
    @staticmethod
    def ensure_owner_or_admin(target_user: User, actor: User) -> None:
        if not actor.is_admin() and str(actor.get_id()) != target_user.get_id():
            raise PermissionError("A user can only access or modify their own account.")

    @staticmethod
    def can_change_role(actor: User, new_role: UserRole | None) -> None:
        if new_role is not None and not actor.is_admin():
            raise PermissionError("Only admins can change user roles.")

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
        UserService.ensure_owner(user, authenticated_user_id)
        user.add_song(song)
