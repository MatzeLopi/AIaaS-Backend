from datetime import datetime, date

from pydantic import BaseModel


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

    # TODO: Adept this to the subscription model.
    disabled: bool
    email_verified: bool
    verification_token: str
    created_at: datetime | None = None
    updated_at: datetime | None = None
    hashed_password: str
    