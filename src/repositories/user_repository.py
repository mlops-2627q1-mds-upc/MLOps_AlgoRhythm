from domain.user import User
from sqlalchemy.orm import Session

class UserRepository:

    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> User | None:
        return db.query(User).filter(User.user_id == user_id).first()

    @staticmethod
    def get_user_by_username(db: Session, username: str) -> User:
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def create_user(db: Session, username: str, hashed_password: str, role: str) -> User:
        db_user = User(username=username, hashed_password=hashed_password, role=role)
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user

    # TODO
    def update_user(user: User) -> User:
        return user

    # TODO
    def delete_user(user: User) -> None:
        return user