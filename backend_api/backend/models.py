from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    UniqueConstraint,
    Index,
    
)


# Custom Imports
from .database import Base


class User(Base):
    """ Model for storing user data.

    Attributes:
        username (str): Username of the user.
        email (str): Email of the user.
        full_name (str): Full name of the user.
        disabled (bool): Indicates if the user is disabled.
        hashed_password (str): Hashed password of the user.
        email_verified (bool): Indicates if the email of the user is verified.
        verification_token (str): Token used for verification.
        created_at (datetime): Date and time when the user was created.
        updated_at (datetime): Date and time when the user was last updated.
        subscription_expiration_date (datetime): Date when the subscription expires.
        subscription_status (str): Status of the subscription.

    """

    __tablename__ = "users"

    username = Column(String(255), primary_key=True, nullable=False)
    email = Column(String(255), unique=True)
    full_name = Column(String(255))
    disabled = Column(Boolean)
    hashed_password = Column(String(255), nullable=False)
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    subscription_expiration_date = Column(DateTime)
    subscription_status = Column(String(50))

    __table_args__ = (
        Index("users_pkey", "username", unique=True),
        Index("idx_users_email", "email"),
        Index("idx_users_username", "username"),
        UniqueConstraint("email", name="users_email_key"),
        UniqueConstraint("username", name="users_username_key"),
    )

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}', full_name='{self.full_name}')>"
