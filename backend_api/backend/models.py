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
    Enum,
    ForeignKey,
)

from sqlalchemy.orm import relationship

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
        organization_id (int): Organization ID of the user.

    Relationship:
        organization (relationship): Organization of the user
        resources (relationship): Resources with permissions for the user
        roles (relationship): Roles of the user

    """

    __tablename__ = "users"

    username = Column(String(255), nullable=False)
    user_id = Column(Integer, primary_key=True, autoincrement="auto")
    email = Column(String(255), unique=True)
    full_name = Column(String(255))
    disabled = Column(Boolean)
    hashed_password = Column(String(255), nullable=False)
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String(255))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    organization_id = Column(String(255), ForeignKey("organizations.organization_id"))

    organization = relationship("Organization", back_populates="users")
    resources = relationship("ResourcePermission", back_populates="user")
    roles = relationship("UserRole", back_populates="user")

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
    user_id = Column(Integer, nullable=False)
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


class TFModel(Base):
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
    user_id = Column(Integer, nullable=False)
    version = Column(Integer, nullable=False)
    model_path = Column(String(255), nullable=False)

    __table_args__ = (
        UniqueConstraint("tf_id", "version", name="unique_model_name_version"),
        Index("indx_user_id", "user_id"),
    )

    def __repr__(self):
        return f"<TFModel(tf_id='{self.tf_id}', tf_name='{self.tf_name}', tf_description='{self.tf_description}', user_id='{self.user_id}', version='{self.version}')>"


class Organization(Base):
    """Organization to which users can belong.


    Attributes:
        organization_id (str): Unique identifier for the organization
        organization_name (str): Name of the organization
        organization_description (str): Description of the organization
        seats (int): Number of seats available in the organization
        cpu_hours (int): Number of CPU hours available in the organization
        gpu_hours (int): Number of GPU hours available in the organization

    Relationship:
        users (relationship): Users belonging to the organization
    """

    __tablename__ = "organizations"

    organization_id = Column(String(255), primary_key=True, nullable=False)
    organization_name = Column(String(255), nullable=False)
    organization_description = Column(Text, nullable=True)
    seats = Column(Integer, nullable=False)
    cpu_hours = Column(Integer, nullable=False)
    gpu_hours = Column(Integer, nullable=False)

    users = relationship("User", back_populates="organization")

    def __repr__(self) -> str:
        return f"<Organization(organization_id='{self.organization_id}', organization_name='{self.organization_name}', organization_description='{self.organization_description}', seats='{self.seats}', cpu_hours='{self.cpu_hours}', gpu_hours='{self.gpu_hours}')>"


class ResourcePermission(Base):
    """Model for storing user permissions on resources.

    Attributes:
        permission_id (int): Unique identifier for the permission
        user_id (int): User ID of the user
        resource_id (str): Unique identifier for the resource
        permission_type (Enum, str): Type of permission (view or edit)

    Relationship:
        user (relationship): User with the permission
        resource (relationship): Resource with the permissionF
    """

    __tablename__ = "resource_permissions"

    permission_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    resource_id = Column(String(255), nullable=False)
    permission_type = Column(
        Enum("view", "edit", name="permission_types"), nullable=False
    )

    user = relationship("User", back_populates="resources")

    def __repr__(self) -> str:
        return f"<ResourcePermission(permission_id='{self.permission_id}', user_id='{self.user_id}', resource_id='{self.resource_id}', permission_type='{self.permission_type}')>"


class Role(Base):
    """Model for storing the roles available.

    Attributes:
        role_id (int): Unique identifier for the role
        role_name (str): Name of the role

    Relationship:
        users (relationship): Users with the role
    """

    __tablename__ = "roles"

    role_id = Column(Integer, primary_key=True, autoincrement=True)
    role_name = Column(String, nullable=False)

    users = relationship("UserRole", back_populates="role")

    def __repr__(self) -> str:
        return f"<Role(role_id='{self.role_id}', role_name='{self.role_name}')>"


class UserRole(Base):
    """Model for handling the roles of a user in an organization.

    Attributes:
        user_role_id (int): Unique identifier for the user role
        user_id (int): User ID of the user
        role_id (int): Role ID of the user
        organization_id (int): Organization ID of the user

    Relationship:
        user (relationship): User with the role
        role (relationship): Role of the user
        organization (relationship): Organization of the user
    """

    __tablename__ = "user_roles"

    user_role_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    role_id = Column(Integer, ForeignKey("roles.role_id"))
    organization_id = Column(String(255), ForeignKey("organizations.organization_id"))

    user = relationship("User", back_populates="roles")
    role = relationship("Role", back_populates="users")
    organization = relationship("Organization")

    def __repr__(self) -> str:
        return f"<UserRole(user_role_id='{self.user_role_id}', user_id='{self.user_id}', role_id='{self.role_id}', organization_id='{self.organization_id}')>"
