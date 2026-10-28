import asyncio
import json
import os
import re
import shutil
import socket
import threading
import time
import webbrowser
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import agent, anki, ocr
from .extract import extract
from .models import PageProposal, note_key

ROOT = Path(__file__).parent.parent
# where the data folder is, if not the default; settings.json lives in the data folder itself
CONFIG = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "language-sketchbook" / "config.json"
EXAMPLES = ROOT / "examples"
STATIC = Path(__file__).parent / "static"
PORT = 8642
BUSY = ("queued", "extracting", "thinking")


def configured_data_dir() -> Path:
    return Path(json.loads(CONFIG.read_text())["data_dir"]) if CONFIG.exists() else ROOT / "data"


DATA = configured_data_dir()
PAGES = DATA / "pages"
SETTINGS = DATA / "settings.json"


def use_data_dir(path: Path) -> None:
    global DATA, PAGES, SETTINGS
    DATA, PAGES, SETTINGS = path, path / "pages", path / "settings.json"
    PAGES.mkdir(parents=True, exist_ok=True)
    seed_demos()


def seed_demos() -> None:
    """Add each example page as a demo page, with its bundled results from examples/<name>/."""
    for image in sorted(EXAMPLES.glob("*")):
        d = PAGES / f"demo-{image.stem}"
        if not image.is_file() or d.exists():
            continue
        processed = EXAMPLES / image.stem
        if processed.is_dir():
            shutil.copytree(processed, d)
        else:
            d.mkdir()
        name = "page" + image.suffix.lower()
        shutil.copy(image, d / name)
        meta = {"id": d.name, "name": image.stem.replace("_", " ").capitalize(), "file": name,
                "status": "ready", "created": 0, "demo": True}
        if not (d / "proposal.json").exists():
            meta["status"] = "queued"  # without bundled results the demo is processed like an upload
        (d / "meta.json").write_text(json.dumps(meta, indent=2))


class Settings(BaseModel):
    model: str = "sonnet"
    ocr_device: str = "gpu"
    anki_url: str = "http://localhost:8765"
    deck_root: str = "Language Sketchbook"


def load_settings() -> Settings:
    s = Settings.model_validate_json(SETTINGS.read_text()) if SETTINGS.exists() else Settings()
    anki.URL = s.anki_url
    if ocr.DEVICE != s.ocr_device:
        ocr.DEVICE = s.ocr_device
        ocr._model.cache_clear()
    return s


# --- pages -------------------------------------------------------------------

events: dict[str, list[dict]] = {}
queue: asyncio.Queue[str] = asyncio.Queue()


def page_dir(page_id: str) -> Path:
    d = PAGES / page_id
    if not re.fullmatch(r"[\w-]+", page_id) or not d.is_dir():
        raise HTTPException(404, "page not found")
    return d


def read_meta(page_id: str) -> dict:
    return json.loads((page_dir(page_id) / "meta.json").read_text())


def write_meta(page_id: str, **changes) -> dict:
    meta = {**read_meta(page_id), **changes}
    (PAGES / page_id / "meta.json").write_text(json.dumps(meta, indent=2))
    emit(page_id, {"type": "status", "status": meta["status"], "error": meta.get("error")})
    return meta


def emit(page_id: str, event: dict) -> None:
    events.setdefault(page_id, []).append(event)


async def worker():
    while True:
        page_id = await queue.get()
        try:
            await process(page_id)
        except Exception as e:
            write_meta(page_id, status="error", error=f"{type(e).__name__}: {e}")
        queue.task_done()


async def process(page_id: str):
    d = page_dir(page_id)
    meta = write_meta(page_id, status="extracting", error=None)
    events[page_id] = [{"type": "status", "status": "extracting"}]
    boxes = await asyncio.to_thread(extract, d / meta["file"], d)
    emit(page_id, {"type": "text", "text": f"Found {len(boxes)} candidate boxes."})
    write_meta(page_id, status="thinking")
    settings = load_settings()
    await agent.run(d, meta["file"], model=settings.model, on_event=lambda e: emit(page_id, e))
    write_meta(page_id, status="ready")


@asynccontextmanager
async def lifespan(app: FastAPI):
    use_data_dir(DATA)
    load_settings()
    # pages interrupted by a restart go back into the queue
    for meta_file in sorted(PAGES.glob("*/meta.json")):
        meta = json.loads(meta_file.read_text())
        if meta["status"] in BUSY:
            queue.put_nowait(meta["id"])
    task = asyncio.create_task(worker())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/icon.svg")
def icon():
    return FileResponse(STATIC / "icon.svg")


@app.get("/settings")
def get_settings() -> Settings:
    return load_settings()


@app.put("/settings")
def put_settings(s: Settings) -> Settings:
    DATA.mkdir(exist_ok=True)
    SETTINGS.write_text(s.model_dump_json(indent=2))
    return load_settings()


class DataDir(BaseModel):
    path: str


@app.get("/data-dir")
def get_data_dir() -> DataDir:
    return DataDir(path=str(DATA))


@app.put("/data-dir")
def put_data_dir(body: DataDir) -> DataDir:
    """Move the data folder to `path`, or switch to it if it already holds app data."""
    new, old = Path(body.path).expanduser(), DATA.resolve()
    if not new.is_absolute():
        raise HTTPException(400, "Use an absolute path.")
    new = new.resolve()
    if new == old:
        return get_data_dir()
    if new.is_relative_to(old):
        raise HTTPException(400, "The new folder can't be inside the current one.")
    if any(m["status"] in BUSY for m in list_pages()):
        raise HTTPException(409, "Wait until all pages are processed.")
    try:
        if not ((new / "pages").is_dir() or (new / "settings.json").exists()):
            if new.exists() and (not new.is_dir() or any(new.iterdir())):
                raise HTTPException(400, f"{new} isn't an empty folder.")
            new.mkdir(parents=True, exist_ok=True)
            for child in old.iterdir():
                shutil.move(child, new / child.name)
        CONFIG.parent.mkdir(parents=True, exist_ok=True)
        CONFIG.write_text(json.dumps({"data_dir": str(new)}, indent=2))
        use_data_dir(new)
    except OSError as e:
        raise HTTPException(400, str(e))
    with suppress(OSError):
        old.rmdir()  # only if it's empty now
    load_settings()
    return get_data_dir()


@app.get("/anki/status")
def anki_status():
    load_settings()
    return anki.status()


@app.post("/anki/install")
def anki_install():
    try:
        anki.ensure_note_types()
    except anki.AnkiError as e:
        raise HTTPException(502, str(e))
    return anki.status()


def page_summary(d: Path) -> dict:
    """meta.json plus what the sidebar shows. The title is the proposal's topic, the same
    name the Anki deck gets; `name` (the upload's file name) only stands in until there is one."""
    meta = json.loads((d / "meta.json").read_text())
    proposal = json.loads((d / "proposal.json").read_text()) if (d / "proposal.json").exists() else {}
    return {**meta, "title": proposal.get("topic") or meta["name"], "lang": proposal.get("target_lang")}


@app.get("/pages")
def list_pages():
    pages = [page_summary(p.parent) for p in PAGES.glob("*/meta.json")]
    return sorted(pages, key=lambda m: m["created"], reverse=True)


@app.post("/pages")
async def upload(files: list[UploadFile]):
    created = []
    for f in files:
        stem = re.sub(r"[^\w-]+", "_", Path(f.filename or "page").stem)[:40]
        page_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}"
        d = PAGES / page_id
        d.mkdir(parents=True)
        name = "page" + (Path(f.filename or "").suffix.lower() or ".jpg")
        (d / name).write_bytes(await f.read())
        meta = {"id": page_id, "name": f.filename, "file": name, "status": "queued", "created": time.time()}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        events[page_id] = [{"type": "status", "status": "queued"}]
        await queue.put(page_id)
        created.append(meta)
    return created


@app.post("/pages/{page_id}/rerun")
async def rerun(page_id: str):
    write_meta(page_id, status="queued", error=None)
    await queue.put(page_id)
    return read_meta(page_id)


@app.get("/pages/{page_id}")
def get_page(page_id: str):
    d = page_dir(page_id)
    result = {"meta": read_meta(page_id), "proposal": None, "existing": [], "commit": None}
    if (d / "proposal.json").exists():
        proposal = PageProposal.model_validate_json((d / "proposal.json").read_text())
        result["proposal"] = proposal.model_dump()
        keys = [note_key(proposal.target_lang, c) for c in proposal.cards]
        result["existing"] = [None] * len(keys)
        try:
            load_settings()
            found = anki.find_existing(keys)
            # existing note per card, in card order
            result["existing"] = [found.get(k) for k in keys]
        except anki.AnkiError as e:
            result["anki_error"] = str(e)
    if (d / "commit.json").exists():
        result["commit"] = json.loads((d / "commit.json").read_text())
    return result


@app.put("/pages/{page_id}")
def save_page(page_id: str, proposal: PageProposal):
    (page_dir(page_id) / "proposal.json").write_text(proposal.model_dump_json(indent=2))
    return get_page(page_id)


@app.post("/pages/{page_id}/commit")
def commit(page_id: str):
    d = page_dir(page_id)
    proposal = PageProposal.model_validate_json((d / "proposal.json").read_text())
    settings = load_settings()
    try:
        results = anki.upsert(proposal, d, settings.deck_root)
    except anki.AnkiError as e:
        raise HTTPException(502, str(e))
    (d / "commit.json").write_text(json.dumps(results, indent=2))
    write_meta(page_id, status="committed")
    return results


@app.get("/pages/{page_id}/files/{name}")
def page_file(page_id: str, name: str):
    path = page_dir(page_id) / name
    if not re.fullmatch(r"[\w.-]+", name) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/pages/{page_id}/thumb")
def page_thumb(page_id: str):
    """A small JPEG of the page for the sidebar, made on first request."""
    thumb = page_dir(page_id) / "thumb.jpg"
    if not thumb.exists():
        img = cv2.imread(str(thumb.parent / read_meta(page_id)["file"]))
        if img is None:
            raise HTTPException(404)
        h, w = img.shape[:2]
        img = cv2.resize(img, (240, max(1, round(h * 240 / w))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(thumb), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return FileResponse(thumb)


@app.get("/pages/{page_id}/events")
async def page_events(page_id: str):
    page_dir(page_id)

    async def stream():
        sent = 0
        while True:
            pending = events.get(page_id, [])
            for e in pending[sent:]:
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
            sent = len(pending)
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


def main():
    url = f"http://localhost:{PORT}"
    # already running (e.g. launched again from the desktop): just open the UI
    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", PORT)) == 0:
            webbrowser.open(url)
            return
    threading.Timer(1.5, webbrowser.open, [url]).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT)
