# FreshTrack

Grocery expiry tracker: photo of a receipt -> OCR -> items -> expiry dates -> reminders.

## Run
    python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env                                  # add your NVIDIA_API_KEY
    uvicorn app:app --reload
Open http://127.0.0.1:8000 (UI) and http://127.0.0.1:8000/docs (API).
Camera capture needs HTTPS on a phone; use a tunnel (e.g. ngrok) or deploy.

## How it works
1. UI compresses the photo and POSTs to `/api/receipts` (returns a job id at once).
2. Server runs local OCR (RapidOCR) and a regex parser. Progress streams over SSE.
3. If parser confidence < `CONFIDENCE_THRESHOLD`, the NVIDIA model structures the text (retry + backoff on 429).
4. Items get category + expiry from `shelf_life.json` (keyword, then fuzzy match).
5. User reviews on the Check screen, then Save writes to SQLite.

## Files
- `app.py` backend (API, OCR, parser, NVIDIA fallback, scheduler)
- `shelf_life.json` shelf-life rules, edit freely
- `static/` frontend (index.html + styles.css)
- `.env.example` copy to `.env`

## Not built yet
User login, real email/push sending (see `check_expiries` in app.py), tests, Postgres.
