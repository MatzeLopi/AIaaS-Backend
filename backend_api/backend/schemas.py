from datetime import datetime

from pydantic import BaseModel, Field
from typing import Optional


class OrganizationBase(BaseModel):
    """Base schema for Organizations

    Attributes:
        organization_name (str): Name of the organization
        organization_description (str): Description of the organization
    """

    organization_name: str
    organization_description: str

    class Config:
        from_attributes = True

class OrganizationFinances(OrganizationBase):
    """Schema for organization finances

    Inherits from OrganizationBase.

    Attributes:
        seats (int): Number of seats in the organization
        cpu_hours (int): Number of CPU hours used by the organization
        gpu_hours (int): Number of GPU hours used by the organization
    """

    seats: int
    cpu_hours: int
    gpu_hours: int

    class Config:
        from_attributes = True


class OrganizationInDB(OrganizationFinances):
    """Schema for organization stored in the database.

    Inherits from OrganizationFinances

    Attributes:
        organization_id (str): Unique identifier for the organization
    """

    organization_id: str

    class Config:
        from_attributes = True


class Role(BaseModel):
    """Model for storing user roles.

    Attributes:
        role_id (int): Unique identifier for the role
        role_name (str): Name of the role
        role_description (str): Description of the role
    """

    role_name: str

    class Config:
        from_attributes = True


class RoleInDB(Role):
    """Schema for role stored in the database.

    Inherits from Role.

    Attributes:
        role_id (int): Unique identifier for the role
    """

    role_id: int

    class Config:
        from_attributes = True


class UserRoles(BaseModel):
    """Model for storing user roles.

    Attributes:
        user_id (int): Unique identifier for the user
        role_id (int): Unique identifier for the role
    """

    user_id: int
    role_id: int
    organization_id: int

    class Config:
        from_attributes = True


class ResourcePermission(BaseModel):
    """Model for storing resource permissions.

    Attributes:
        user_id (int): Unique identifier for the user
        resource_id (int): Unique identifier for the resource
        permission_type (str): Type of the permission
    """

    user_id: int
    resource_id: int
    permission_type: str

    class Config:
        from_attributes = True


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

class UserOrganization(UserBase):
    """Schema for creating a new user.

    Inherits from UserBase.

    Attributes:

    """
    organization_id: str

    class Config:
        from_attributes = True

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
    organization_id: str

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
        from_attributes = True


class TFBase(BaseModel):
    """Base schema for model output to users.

    Attributes:
        model_name (str): Name of the model.
        model_description (str, Optional): Description of the model.
        version (int): Version of the model.
    """

    tf_name: str = Field(..., max_length=255)
    tf_description: Optional[str]
    version: int

    class Config:
        from_attributes = True


class TFInDB(TFBase):
    """Schema for model stored in the database.

    Attributes:
        model_id (str): Unique identifier for the model.
        created_at (datetime): Date and time when the model was created.
        updated_at (datetime): Date and time when the model was last updated.
        user_id (str): User ID of the model owner

    """

    tf_id: str = Field(..., max_length=255)
    user_id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
