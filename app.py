from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ai import ask_ai

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()

    user_message = data.get("message", "").strip()

    if not user_message:
        return JSONResponse(
            {"response": "Please enter a message."},
            status_code=400
        )

    try:
        reply = ask_ai(user_message)

        return JSONResponse({
            "response": reply
        })

    except Exception as e:
        return JSONResponse(
            {"response": f"Error: {str(e)}"},
            status_code=500
        )