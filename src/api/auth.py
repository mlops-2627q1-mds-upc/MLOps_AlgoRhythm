from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from src.database import get_db
from src.domain.user import User

def get_current_user(db: Session = Depends(get_db)) -> User:
    user = db.query(User).first()
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return user

def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_admin():
        raise HTTPException(
            status_code=403, 
            detail="You do not have permission to access this resource."
        )
    return current_user