from datetime import datetime, date

from pydantic import BaseModel, Field
from typing import Optional


class UserBase(BaseModel):
    """Base schema for the user.

    Attributes:
        username (str): Username of the user.
        email (str): Email of the user.
        full_name (str, Optional): Full name of the user.
    """

    username: str
    email: str
    full_name: str | None = None

    class Config:
        from_attributes = True


class UserCreate(UserBase):
    """Schema for creating a new user.

    Includes the password of the user.
    This schema is never saved.

    Inherits from UserBase.

    Attributes:
        password (str): Password of the user.
    """

    password: str


class UserInDB(UserBase):
    """Schema for the user in the database.

    Inherits from UserBase.

    Attributes:
        disabled (bool): Indicates if the user is disabled.
        email_verified (bool): Indicates if the email of the user is verified.
        verification_token (str): Token used for verification.
        created_at (datetime, Optional): Date and time when the user was created.
        updated_at (datetime, Optional): Date and time when the user was updated.
        hashed_password (str): Hashed password of the user.
    """

    disabled: bool
    email_verified: bool
    verification_token: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    hashed_password: str
    subscription_expiration_date: Optional[datetime] = None
    subscription_status: Optional[str] = None

    class Config:
        from_attributes = True


class DatasetBase(BaseModel):
    """Base schema for dataset output to users.

    Attributes:
        dataset_name (str, Optional): Name of the dataset.
        dataset_description (str, Optional): Description of the dataset.
        dataset_type (str): Type of the dataset.
        version (int): Version of the dataset
    """

    dataset_name: Optional[str] = Field(None, max_length=255)
    dataset_description: Optional[str] = Field(None, max_length=255)
    dataset_type: str = Field(..., max_length=255)
    version: int

    class Config:
        from_attributes = True


class DatasetInDB(DatasetBase):
    """Schema for dataset stored in the database.

    Attributes:
        dataset_id (str): Unique identifier for the dataset.
        created_at (datetime): Date and time when the dataset was created.
        updated_at (datetime): Date and time when the dataset was last updated.
        user_id (str): User ID of the dataset owner.

    """

    dataset_id: str = Field(..., max_length=255)
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True


class ModelBase(BaseModel):
    """Base schema for model output to users.

    Attributes:
        model_name (str): Name of the model.
        model_description (str, Optional): Description of the model.
        version (int): Version of the model.
    """

    model_name: str = Field(..., max_length=255)
    model_description: Optional[str]
    version: int

    class Config:
        from_attributes = True


class ModelInDB(ModelBase):
    """Schema for model stored in the database.

    Attributes:
        model_id (str): Unique identifier for the model.
        created_at (datetime): Date and time when the model was created.
        updated_at (datetime): Date and time when the model was last updated.
        user_id (str): User ID of the model owner

    """

    model_id: str = Field(..., max_length=255)
    user_id = Field(..., max_length=255)
    created_at: datetime
    updated_at: datetime

    class Config:
        orm_mode = True
