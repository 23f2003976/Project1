from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse
from tasks import execute_task, read_file
from pathlib import Path
from typing import Optional
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import traceback
import re
import os
import uvicorn

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allow all origins (you can specify a list of allowed origins here)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
BASE_DATA_DIR = Path("/data").resolve()


class QuestionRequest(BaseModel):
    question: str


@app.post("/run")
async def run_task(request: Request,
                   question: Optional[str] = Query(None, min_length=1)):
    q = question

    if not q:
        try:
            body = await request.json()
            q = body.get("question")
        except Exception:
            pass

    if not q or len(q) < 1:
        raise HTTPException(status_code=422,
                            detail="Question parameter is required")
    lowered = q.lower()
    forbidden_words = {"delete", "remove", "rm", "truncate", "unlink"}
    words = re.findall(r"\b\w+\b", lowered)
    if any(word in forbidden_words for word in words):
        raise HTTPException(status_code=400,
                            detail="Task contains unsafe delete instructions.")
    try:
        result = await execute_task(q)
        return {"status": "success", "answer": result, "links": []}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500,
                            detail=f"Internal Server Error: {str(e)}")


@app.get("/read")
async def read(path: str):
    try:
        if not str(Path(path)).startswith(str(BASE_DATA_DIR)):
            raise HTTPException(status_code=403,
                                detail="Access outside /data is forbidden")
        target_path = (BASE_DATA_DIR / path).resolve()
        content = await read_file(str(target_path))
        return PlainTextResponse(content)
    except FileNotFoundError:
        traceback.print_exc()
        return PlainTextResponse("", status_code=404)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))  # Use Render's PORT or default to 3000 locally
    uvicorn.run(app, host="0.0.0.0", port=port)
