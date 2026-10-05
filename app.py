"""FreshTrack backend: receipt OCR -> parse -> shelf life -> expiry tracking.

Run:  uvicorn app:app --reload
Docs: http://127.0.0.1:8000/docs
"""
import asyncio, io, json, logging, os, re, sqlite3, time, uuid
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from rapidfuzz import fuzz, process

load_dotenv()
log = logging.getLogger("freshtrack")
logging.basicConfig(level=logging.INFO)

DB_PATH = os.getenv("DB_PATH", "freshtrack.db")
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))
NVIDIA_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = os.getenv("NVIDIA_TEXT_MODEL", "meta/llama-3.1-8b-instruct")
MAX_UPLOAD = 5 * 1024 * 1024

# Shelf-life rules live in memory for sub-5ms lookups.
RULES = json.loads((Path(__file__).parent / "shelf_life.json").read_text())
KEYWORDS = {kw: r for r in RULES for kw in r["keywords"]}
NAME_CACHE: dict[str, str] = {}

app = FastAPI(title="FreshTrack")


# ---------------------------------------------------------------- database
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS items(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          name TEXT NOT NULL, category TEXT, quantity REAL DEFAULT 1,
          price REAL, storage TEXT DEFAULT 'fridge',
          purchase_date TEXT, expiry_date TEXT,
          status TEXT DEFAULT 'active', created_at TEXT DEFAULT CURRENT_TIMESTAMP);
        CREATE INDEX IF NOT EXISTS idx_items_expiry ON items(status, expiry_date);
        """)


# ---------------------------------------------------------------- models
class ItemIn(BaseModel):
    name: str
    category: str | None = None
    quantity: float = 1
    price: float | None = None
    storage: str = "fridge"
    purchase_date: str | None = None
    expiry_date: str | None = None


class ItemPatch(BaseModel):
    name: str | None = None
    category: str | None = None
    quantity: float | None = None
    storage: str | None = None
    expiry_date: str | None = None
    status: str | None = None  # active | consumed | discarded


# ---------------------------------------------------------------- shelf life
def find_rule(name: str):
    """Keyword match first, then fuzzy match. Returns rule or None."""
    key = name.lower()
    if key in NAME_CACHE:
        return KEYWORDS.get(NAME_CACHE[key])
    for word in re.findall(r"[a-z]+", key):
        if word in KEYWORDS:
            NAME_CACHE[key] = word
            return KEYWORDS[word]
    best = process.extractOne(key, KEYWORDS.keys(), scorer=fuzz.partial_ratio, score_cutoff=85)
    if best:
        NAME_CACHE[key] = best[0]
        return KEYWORDS[best[0]]
    return None


def compute_expiry(name: str, purchase: str, storage: str = "fridge"):
    rule = find_rule(name)
    if not rule:
        return None, None
    days = rule["days_fridge"] if storage == "fridge" else rule["days_room"]
    base = datetime.strptime(purchase, "%Y-%m-%d").date()
    return rule["category"], (base + timedelta(days=days)).isoformat()


def decorate(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d["expiry_date"]:
        left = (datetime.strptime(d["expiry_date"], "%Y-%m-%d").date() - date.today()).days
        d["days_left"] = left
        d["state"] = "expired" if left < 0 else "soon" if left <= 2 else "fresh"
    else:
        d["days_left"], d["state"] = None, "unknown"
    return d


# ---------------------------------------------------------------- OCR + parsing
_ocr = None


def ocr_text(image_bytes: bytes) -> str:
    global _ocr
    from PIL import Image
    import numpy as np
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((1500, 1500))
    result, _ = _ocr(np.array(img))
    return "\n".join(r[1] for r in (result or []))


DATE_PATTERNS = [
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", "%d-%m-%Y"),
    (r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2})\b", "%d-%m-%y"),
]
LINE_RE = re.compile(r"^(?P<name>[A-Za-z][A-Za-z0-9 .&'%/-]{2,}?)\s+(?:(?P<qty>\d+(?:\.\d+)?)\s*[xX]?\s+)?(?P<price>\d+\.\d{2})\s*$")
SKIP = re.compile(r"total|tax|gst|cgst|sgst|change|cash|upi|invoice|bill|mrp|discount", re.I)


def parse_receipt(text: str) -> dict:
    purchase = None
    for pat, fmt in DATE_PATTERNS:
        m = re.search(pat, text)
        if m:
            try:
                purchase = datetime.strptime("-".join(m.groups()), fmt).date().isoformat()
                break
            except ValueError:
                pass
    items = []
    for line in text.splitlines():
        line = line.strip()
        m = LINE_RE.match(line)
        if m and not SKIP.search(line):
            items.append({"name": m["name"].strip(), "quantity": float(m["qty"] or 1), "price": float(m["price"])})
    known = sum(1 for i in items if find_rule(i["name"]))
    conf = (known / len(items)) if items else 0.0
    if not purchase:
        conf *= 0.6
    return {"store": text.splitlines()[0].strip() if text else "", "purchase_date": purchase,
            "items": items, "confidence": round(conf, 2)}


def nvidia_parse(text: str) -> dict | None:
    """Fallback: ask the NVIDIA-hosted model to structure messy OCR text."""
    if not NVIDIA_KEY:
        return None
    from openai import OpenAI
    client = OpenAI(base_url=NVIDIA_URL, api_key=NVIDIA_KEY, timeout=6)
    prompt = ("Extract this grocery receipt as JSON only, no prose: "
              '{"store":"","purchase_date":"YYYY-MM-DD","items":[{"name":"","quantity":1,"price":0}]}\n\n' + text)
    for attempt in range(3):
        try:
            r = client.chat.completions.create(model=NVIDIA_MODEL, temperature=0, max_tokens=900,
                                               messages=[{"role": "user", "content": prompt}])
            raw = re.sub(r"^```(?:json)?|```$", "", r.choices[0].message.content.strip(), flags=re.M).strip()
            return json.loads(raw)
        except Exception as e:  # 429s, timeouts, bad JSON
            log.warning("NVIDIA attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return None


def process_receipt(image_bytes: bytes) -> dict:
    text = ocr_text(image_bytes)
    parsed = parse_receipt(text)
    if parsed["confidence"] < THRESHOLD:
        better = nvidia_parse(text)
        if better and better.get("items"):
            parsed = {**parsed, **better, "confidence": 0.8, "used_llm": True}
    purchase = parsed.get("purchase_date") or date.today().isoformat()
    for it in parsed["items"]:
        cat, exp = compute_expiry(it["name"], purchase)
        it.update(category=cat or "Unknown", expiry_date=exp, storage="fridge")
    parsed["purchase_date"] = purchase
    return parsed


# ---------------------------------------------------------------- jobs + SSE
JOBS: dict[str, asyncio.Queue] = {}


async def run_job(job_id: str, image_bytes: bytes):
    q = JOBS[job_id]
    loop = asyncio.get_running_loop()
    try:
        await q.put({"stage": "uploaded"})
        await q.put({"stage": "reading"})
        result = await loop.run_in_executor(None, process_receipt, image_bytes)
        await q.put({"stage": "ready", "result": result})
    except Exception as e:
        log.exception("job failed")
        await q.put({"stage": "error", "message": f"Could not read receipt: {e}"})
    await q.put(None)


@app.post("/api/receipts")
async def upload_receipt(bg: BackgroundTasks, file: UploadFile = File(...)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Upload a JPG, PNG or WebP image.")
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 5 MB.")
    job_id = uuid.uuid4().hex
    JOBS[job_id] = asyncio.Queue()
    bg.add_task(run_job, job_id, data)
    return {"job_id": job_id}  # returns immediately


@app.get("/api/receipts/{job_id}/events")
async def receipt_events(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        raise HTTPException(404, "Unknown job.")

    async def stream():
        while (ev := await q.get()) is not None:
            yield f"data: {json.dumps(ev)}\n\n"
        JOBS.pop(job_id, None)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ---------------------------------------------------------------- items API
@app.get("/api/items")
def list_items(status: str = "active", limit: int = 50, offset: int = 0):
    with db() as con:
        rows = con.execute(
            "SELECT * FROM items WHERE status=? ORDER BY expiry_date IS NULL, expiry_date LIMIT ? OFFSET ?",
            (status, limit, offset)).fetchall()
    return [decorate(r) for r in rows]


@app.post("/api/items", status_code=201)
def add_items(items: list[ItemIn]):
    """Accepts one or many items (used by the review screen's Save button)."""
    today = date.today().isoformat()
    with db() as con:
        for it in items:
            purchase = it.purchase_date or today
            cat, exp = it.category, it.expiry_date
            if not exp:
                c2, exp = compute_expiry(it.name, purchase, it.storage)
                cat = cat or c2
            con.execute("INSERT INTO items(name,category,quantity,price,storage,purchase_date,expiry_date) VALUES(?,?,?,?,?,?,?)",
                        (it.name, cat, it.quantity, it.price, it.storage, purchase, exp))
    return {"saved": len(items)}


@app.patch("/api/items/{item_id}")
def update_item(item_id: int, patch: ItemPatch):
    fields = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(400, "Nothing to update.")
    sets = ", ".join(f"{k}=?" for k in fields)
    with db() as con:
        cur = con.execute(f"UPDATE items SET {sets} WHERE id=?", (*fields.values(), item_id))
    if cur.rowcount == 0:
        raise HTTPException(404, "Item not found.")
    return {"updated": item_id}


@app.delete("/api/items/{item_id}")
def delete_item(item_id: int):
    with db() as con:
        con.execute("DELETE FROM items WHERE id=?", (item_id,))
    return {"deleted": item_id}


@app.get("/health")
def health():
    return {"ok": True, "nvidia_configured": bool(NVIDIA_KEY)}


# ---------------------------------------------------------------- middleware, scheduler
@app.middleware("http")
async def timing(request: Request, call_next):
    t = time.perf_counter()
    resp = await call_next(request)
    ms = (time.perf_counter() - t) * 1000
    resp.headers["X-Response-Time-ms"] = f"{ms:.1f}"
    if ms > 300 and not request.url.path.startswith("/api/receipts"):
        log.warning("slow request %s %.0fms", request.url.path, ms)
    return resp


def check_expiries():
    """Daily job: find items expiring tomorrow. Plug email/push sending in here."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    with db() as con:
        rows = con.execute("SELECT name FROM items WHERE status='active' AND expiry_date=?", (tomorrow,)).fetchall()
    if rows:
        log.info("Expiring tomorrow: %s", ", ".join(r["name"] for r in rows))


@app.on_event("startup")
def startup():
    init_db()
    from apscheduler.schedulers.background import BackgroundScheduler
    sched = BackgroundScheduler()
    sched.add_job(check_expiries, "cron", hour=8)
    sched.start()


# Serve the UI last so /api routes take priority.
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
