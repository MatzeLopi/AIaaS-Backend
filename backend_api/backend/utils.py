from random import SystemRandom
from string import ascii_uppercase, digits
from pathlib import Path
from time import sleep
from functools import wraps

from fastapi_mail import FastMail, MessageSchema, MessageType, ConnectionConfig
from fastapi.responses import JSONResponse
from pydantic import EmailStr

from passlib.context import CryptContext

from rapidjson import loads

# TODO: Change this to use environment variables
try:
    file_path = Path(Path(__file__).parent.absolute(), "config.json")
    with open(file_path, "r") as f:
        data = f.read()
        CONFIG = loads(data)
except FileNotFoundError as e:
    print(f"Config file not found: {e}")
    print(file_path)
    raise SystemExit

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def generate_verification_token() -> str:
    """Generate a verification token.

    Returns:
        str: Verification token.
    """
    return "".join(SystemRandom().choice(ascii_uppercase + digits) for _ in range(15))


async def get_password_hash(password: str) -> str:
    """Create a password hash.

    Args:
        password (str): Password to hash.

    Returns:
        str: Hashed password.
    """
    return pwd_context.hash(password)


async def send_email(email: EmailStr, body: str, subject: str):
    """Send an email to the user.

    Args:
        email (EmailStr): Email address of the user.
        background_tasks (BackgroundTasks): Background tasks object to run the task asynchronously.

    Returns:
        JSONResponse: Response object.
    """
    message = MessageSchema(
        subject=subject,
        recipients=[email],
        body=body,
        subtype=MessageType.plain,
    )

    connnection_conf = ConnectionConfig(**CONFIG["mail_conf"])

    fm = FastMail(connnection_conf)
    await fm.send_message(message)
    return JSONResponse(status_code=200, content={"message": "email has been sent"})


def retry(
    exceptions: tuple[Exception], tries: int = 3, delay: int = 2, backoff: int = 2
):
    """Retry decorator.

    Args:
        exceptions (tuple): The exceptions to catch.
        tries (int, optional): Number of tries. Defaults to 3.
        delay (int, optional): Delay between retries. Defaults to 2.
        backoff (int, optional): Backoff factor. Defaults to 2.
    """

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions:
                    sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            return f(*args, **kwargs)

        return wrapper

    return decorator
