from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ChatRequest(BaseModel):
    message: str

def validate_message(message: str):
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Empty message not allowed")
    if len(message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long (max 2000 chars)")

async def call_gemini(message: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": message}]}]}

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

@app.post("/chat")
async def chat(request: ChatRequest):
    message = request.message
    validate_message(message)
    reply = await call_gemini(message)
    print(f"User : {message}")
    print(f"AI   : {reply}")
    print("-" * 50)
    return {"reply": reply}

@app.get("/")
def serve_ui():
    return FileResponse("index.html")
