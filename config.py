import os
import sys
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("MODEL", "openai/gpt-oss-20b:free")

if not OPENROUTER_API_KEY:
    sys.exit("FATAL: OPENROUTER_API_KEY is not set. Check your .env file.")