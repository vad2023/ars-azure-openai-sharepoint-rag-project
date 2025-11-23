import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    TENANT_ID = os.getenv("TENANT_ID")
    SP_CLIENT_ID = os.getenv("SP_CLIENT_ID")
    SP_CLIENT_SECRET = os.getenv("SP_CLIENT_SECRET")
    SP_SITE_HOSTNAME = os.getenv("SP_SITE_HOSTNAME")
    SP_SITE_PATH = os.getenv("SP_SITE_PATH")
    SP_DOC_LIBRARY_NAME = os.getenv("SP_DOC_LIBRARY_NAME")

    AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
    AOAI_API_KEY = os.getenv("AOAI_API_KEY")
    AOAI_CHAT_MODEL = os.getenv("AOAI_CHAT_MODEL")
    AOAI_EMBED_MODEL = os.getenv("AOAI_EMBED_MODEL")

    @classmethod
    def validate(cls):
        missing = [
            k for k, v in cls.__dict__.items()
            if not k.startswith("_") and not callable(v) and v is None
        ]
        if missing:
            raise RuntimeError(f"Missing configuration values: {missing}")
