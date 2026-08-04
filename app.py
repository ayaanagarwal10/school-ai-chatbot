import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai import ask_ai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return JSONResponse({"response": "Please enter a message."}, status_code=400)

    if len(user_message) > 1000:
        return JSONResponse(
            {"response": "Message too long. Please keep it under 1000 characters."},
            status_code=400,
        )

    try:
        reply = await ask_ai(user_message)
        return JSONResponse({"response": reply})
    except Exception:
        logger.exception("ask_ai failed")
        return JSONResponse(
            {"response": "Something went wrong. Please try again shortly."},
            status_code=500,
        )