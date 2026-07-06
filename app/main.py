from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.schemas.chat import ChatRequest
from app.schemas.detection import DetectionRequest
from app.services.chat_service import ChatService
from app.services.mtscid_detector import MtsCIDDetector
from app.services.session_store import SessionStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / 'app' / 'static'

app = FastAPI(title='TSAD Agent', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)
app.mount('/static', StaticFiles(directory=str(STATIC_DIR)), name='static')

store = SessionStore()
detector = MtsCIDDetector()
chat_service = ChatService(store)


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/api/health')
def health():
    return {'status': 'ok'}


@app.get('/api/datasets')
def datasets():
    return {'datasets': detector.available_datasets()}


@app.post('/api/detect')
def detect(request: DetectionRequest):
    try:
        record = detector.detect(request)
        store.save_result(record)
        return record
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get('/api/results/{result_id}')
def result(result_id: str):
    record = store.get_result(result_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f'Unknown result_id: {result_id}')
    return record


@app.post('/api/chat')
def chat(request: ChatRequest):
    try:
        return chat_service.answer(request)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
