from datetime import datetime, date

from pydantic import BaseModel


class StockBase(BaseModel):
    """Base schema for stock data.

    Attributes:
        ticker (str): Ticker of the stock.
        date (date): Date when the stock data was recorded.
        open (float): Opening price of the stock.
        high (float): Highest price of the stock on that day.
        low (float): Lowest price of the stock on that day.
        close (float): Closing price of the stock.
        volume (int): Volume of the stock.
    """

    ticker: str
    date: date
    opening: float
    high: float
    low: float
    closing: float
    volume: int

    class Config:
        from_attributes = True




class StockMetaBase(BaseModel):
    """Base schema for stock meta data.

    Attrributes:
        ticker (str): Ticker of the stock.
        name (str): Name of the stock.
        sector (str): Sector of the stock.
        next_earnings (date): Date of the next earnings for the stock.
    """

    ticker: str
    name: str
    sector: str
    next_earnings: date

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

class RedditScore(BaseModel):
    """ Schema for the Reddit score.
    Reddit score reflects the sentiment of the stock on Reddit.

    Can range from 0 (brarish) to 10 (bullish).

    Attributes:
        score (int): Reddit score.
        date (date): Date when the score was recorded
    """
    score:int
    date: date
    
    class Config:
        from_attributes = True

class RedditData(BaseModel):
    """ Schema for the Reddit data.
    Reddit data contains the sentiment of the stock on Reddit.

    Attributes:
        date (date): Date when the score was recorded
        title (str): Title of the Reddit post.
        content(str): Content of the Reddit post.
    """
    date: date
    title: str
    content: str
    
    class Config:
        from_attributes = True