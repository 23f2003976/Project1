from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import PlainTextResponse
from tasks import execute_task, read_file
from pathlib import Path
import traceback
import re

app = FastAPI()

BASE_DATA_DIR = Path("/data").resolve()

@app.post("/run")
async def run_task(task: str = Query(..., min_length=1), user_email: str = Query(...)):
    lowered = task.lower()
    forbidden_words = {"delete", "remove", "rm", "truncate", "unlink"}
    # Extract words, e.g., "format" -> ["format"]
    words = re.findall(r"\b\w+\b", lowered)
    if any(word in forbidden_words for word in words):
        raise HTTPException(status_code=400, detail="Task contains unsafe delete instructions.")
    try:
        result = await execute_task(task, user_email)
        return {"status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.get("/read")
async def read(path: str):
    try:
        if not str(Path(path)).startswith(str(BASE_DATA_DIR)):
            raise HTTPException(status_code=403, detail="Access outside /data is forbidden")
        target_path = (BASE_DATA_DIR / path).resolve()
        content = await read_file(str(target_path))
        return PlainTextResponse(content)
    except FileNotFoundError:
        traceback.print_exc()
        return PlainTextResponse("", status_code=404)

