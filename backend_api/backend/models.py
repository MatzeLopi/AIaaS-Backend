from datetime import datetime

from sqlalchemy import (
    Column,
    String,
    Boolean,
    DateTime,
    UniqueConstraint,
    Index,
    Integer,
    Text,
)


# Custom Imports
from .database import Base


class User(Base):
    """Model for storing user data.

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


class Datasets(Base):
    """Model for storing dataset information

    Attributes:
        dataset_id (str): Unique identifier for the dataset
        dataset_name (str): Name of the dataset
        dataset_description (str): Description of the dataset
        created_at (datetime): Date and time when the dataset was created
        updated_at (datetime): Date and time when the dataset was last updated
        user_id (str): User ID of the dataset owner
        dataset_type (str): Type of the dataset
        version (int): Version of the dataset

    """

    __tablename__ = "datasets"

    dataset_id = Column(String(255), primary_key=True, nullable=False)
    dataset_name = Column(String(255), nullable=True)
    dataset_description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    user_id = Column(String(255), nullable=False)
    dataset_type = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False)
    dataset_path = Column(String(255), nullable=False)

    __table_args__ = (
        UniqueConstraint("dataset_name", name="unique_dataset_name"),
        Index("idx_user_id", "user_id"),
        Index("idx_dataset_type", "dataset_type"),
    )

    def __repr__(self):
        return f"<Dataset(dataset_id='{self.dataset_id}', dataset_name='{self.dataset_name}', dataset_description='{self.dataset_description}', user_id='{self.user_id}', dataset_type='{self.dataset_type}')>"


class TFModels(Base):
    """Model for storing model information

    Attributes:
        tf_id (str): Unique identifier for the model
        tf_name (str): Name of the model
        tf_description (str): Description of the model
        created_at (datetime): Date and time when the model was created
        updated_at (datetime): Date and time when the model was last updated
        user_id (str): User ID of the model owner
        version (int): Version of the model
    """

    __tablename__ = "tf_models"

    tf_id = Column(String(255), primary_key=True, nullable=False)
    tf_name = Column(String(255), nullable=False)
    tf_description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.now, onupdate=datetime.now, nullable=False
    )
    user_id = Column(String(255), nullable=False)
    version = Column(Integer, nullable=False)
    model_path = Column(String(255), nullable=False)

    __table_args__ = (
        UniqueConstraint("tf_id", "version", name="unique_model_name_version"),
        Index("indx_user_id", "user_id"),
    )
