# main.py
import os
import random
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import google.generativeai as genai

# --- Config ---
GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
if not os.getenv("GOOGLE_API_KEY"):
    raise RuntimeError("Missing GOOGLE_API_KEY")

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# --- App ---
app = FastAPI(title="Gemini Q&A + Giggle Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    system_prompt: Optional[str] = "You are a concise, helpful assistant."

class AskResponse(BaseModel):
    answer: str
    funny_addon: str
    combined: str

# --- Funny Addons ---
FUNNY_TAGLINES = [
    "P.S. I asked a squirrel for feedback. It said 'nuts about it!'",
    "Disclaimer: I was trained by cats, so occasional meows may appear.",
    "Fun fact: this response contains zero calories.",
    "I also consulted a goldfish, it promptly forgot.",
    "Warning: may cause uncontrollable nodding.",
]

def pick_funny() -> str:
    return random.choice(FUNNY_TAGLINES)

# --- Routes ---
@app.post("/ask", response_model=AskResponse)
async def ask(data: AskRequest):
    try:
        model = genai.GenerativeModel(
            model_name=GENAI_MODEL,
            system_instruction=data.system_prompt
        )
        resp = model.generate_content([{"role": "user", "parts": [data.question]}])
        answer = (resp.text or "").strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini error: {e}")

    funny = pick_funny()
    combined = f"{answer}\n\n{funny}"
    return AskResponse(answer=answer, funny_addon=funny, combined=combined)

@app.get("/healthz")
async def healthz():
    return {"ok": True, "model": GENAI_MODEL}
