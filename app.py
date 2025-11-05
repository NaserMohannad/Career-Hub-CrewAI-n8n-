
import os, pathlib, time, json
from typing import Any, Dict

os.environ.setdefault("HOME", "/home/user")
os.environ.setdefault("XDG_DATA_HOME", "/home/user/.local/share")
os.environ.setdefault("CREWAI_STORAGE_PATH", "/home/user/.local/share/crewai")

pathlib.Path(os.environ["XDG_DATA_HOME"]).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.environ["CREWAI_STORAGE_PATH"]).mkdir(parents=True, exist_ok=True)
pathlib.Path("./incoming").mkdir(parents=True, exist_ok=True)
pathlib.Path("./daily-job-recommendations").mkdir(parents=True, exist_ok=True)

from fastapi import FastAPI, Request, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    from pipeline import run_pipeline, INCOMING_DIR, OUTPUT_DIR
except Exception as e:
    run_pipeline = None
    INCOMING_DIR = "./incoming"
    OUTPUT_DIR = "./daily-job-recommendations"
    PIPELINE_IMPORT_ERROR = str(e)
else:
    PIPELINE_IMPORT_ERROR = None

APP_START_TS = time.strftime("%Y-%m-%d %H:%M:%S")

app = FastAPI(title="Nasser Job Scraper", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ضيّقها لاحقًا على دومين n8n لو حاب
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

def _save_payload_to_file(data: Dict[str, Any]) -> str:
    pathlib.Path(INCOMING_DIR).mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    path = f"{INCOMING_DIR}/payload_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path

@app.get("/", response_class=PlainTextResponse)
def root():
    return (
        "✅ Nasser Job Scraper is running.\n"
        f"Started at: {APP_START_TS}\n"
        "Endpoints:\n"
        "  GET  /health\n"
        "  POST /webhook   (receive JSON and run pipeline in background)\n"
        "  POST /run       (run pipeline manually; ?sync=true for blocking)\n"
    )

@app.get("/health")
def health():
    return {
        "ok": True,
        "started_at": APP_START_TS,
        "pipeline_import_error": PIPELINE_IMPORT_ERROR,
        "incoming_dir": INCOMING_DIR,
        "output_dir": OUTPUT_DIR,
    }

@app.post("/webhook")
async def webhook(request: Request, background: BackgroundTasks):
    # لوج تشخيصي خفيف
    try:
        h = dict(request.headers)
        print("[/webhook] method=", request.method)
        print("[/webhook] content-type=", h.get("content-type"))
        raw = await request.body()
        print("[/webhook] body_len=", len(raw))
    except Exception:
        pass

    ct = (request.headers.get("content-type") or "").lower()
    payload = None

    # نقبل أي "application/json" حتى لو مع charset
    if "application/json" in ct:
        try:
            payload = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
    else:
        # fallback: حاول نفك JSON من البودي الخام
        try:
            txt = (raw or b"").decode("utf-8", errors="ignore")
            payload = json.loads(txt) if txt else {}
        except Exception:
            raise HTTPException(status_code=415, detail="Unsupported Content-Type; expected JSON.")

    # لو اجانا سترنغ JSON (شائع مع n8n Raw Body) حوّله لقاموس
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Body is a string that isn't valid JSON: {e}")

    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Payload must be a JSON object (not list/scalar).")

    # تطبيع الحقول المهمة للبايبلاين
    interests = payload.get("interests", "")
    if isinstance(interests, list):
        payload["interests"] = ", ".join([str(x).strip() for x in interests if str(x).strip()])
    elif not isinstance(interests, str):
        payload["interests"] = ""
    if not payload["interests"].strip():
        raise HTTPException(status_code=422, detail="Field 'interests' is required (string or list).")

    saved_path = _save_payload_to_file(payload)

    if run_pipeline is None:
        return JSONResponse(status_code=500, content={
            "ok": False, "error": "Pipeline import failed",
            "details": PIPELINE_IMPORT_ERROR, "saved": saved_path
        })

    background.add_task(run_pipeline)
    return {"ok": True, "message": "Payload received. Pipeline scheduled.", "saved": saved_path, "keys": list(payload.keys()), "output_dir": OUTPUT_DIR}

@app.post("/run")
def run_now(background: BackgroundTasks, sync: bool = False):
    if run_pipeline is None:
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": "Pipeline import failed", "details": PIPELINE_IMPORT_ERROR},
        )
    if sync:
        try:
            run_pipeline()
            return {"ok": True, "ran": "sync", "output_dir": OUTPUT_DIR}
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
    else:
        background.add_task(run_pipeline)
        return {"ok": True, "ran": "background", "output_dir": OUTPUT_DIR}
