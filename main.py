"""
YouTube Transcript API Server
FastAPI service wrapping yt-dlp + youtube-transcript-api for subtitle extraction,
with OpenRouter Gemini fallback for audio transcription.
"""

import os
import re
import json
import base64
import logging
import tempfile
import subprocess
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import yt_dlp


# ── .env loader ──────────────────────────────────────────────────────────────

def _load_dotenv():
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value

_load_dotenv()


# ── Config ───────────────────────────────────────────────────────────────────

VALID_API_KEY = os.environ.get("CCSEARCH_API_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_MODEL = "google/gemini-3-flash-preview"
CHUNK_MB_MAX = 18

ZH_CODES = frozenset({"zh-Hant", "zh-TW", "zh-Hans", "zh-CN", "zh"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("yt-transcript")


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title="YouTube Transcript API", docs_url=None, redoc_url=None)


@app.exception_handler(HTTPException)
async def custom_http_exception(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": str(exc.detail)},
    )


# ── Request Model ────────────────────────────────────────────────────────────

class TranscriptRequest(BaseModel):
    url: str
    lang: str = "zh-Hant"
    timestamps: bool = False
    format: str = "text"


# ── Video ID Extraction ─────────────────────────────────────────────────────

def extract_video_id(url_or_id: str) -> str:
    s = url_or_id.strip()
    if re.fullmatch(r"[\w-]{11}", s):
        return s
    m = re.search(r"youtu\.be/([\w-]{11})", s)
    if m:
        return m.group(1)
    m = re.search(r"(?:v=|/v/|/embed/|/shorts/)([\w-]{11})", s)
    if m:
        return m.group(1)
    raise ValueError(f"Cannot extract video ID from: {s}")


# ── Language Matching ────────────────────────────────────────────────────────

def lang_matches(avail: str, target: str) -> bool:
    if avail == target:
        return True
    if target in ZH_CODES:
        if avail in ZH_CODES:
            return True
        return any(avail.startswith(zc + "-") or avail == zc for zc in ZH_CODES)
    if avail == target:
        return True
    if avail.startswith(target + "-"):
        return True
    return False


def find_lang_key(subs: dict, target: str) -> str | None:
    if not subs:
        return None
    if target in subs:
        return target
    if target in ZH_CODES:
        for zc in ZH_CODES:
            if zc in subs:
                return zc
    for key in subs:
        if lang_matches(key, target):
            return key
    return None


def find_fallback_key(subs: dict, prefer: str = "en") -> str | None:
    if not subs:
        return None
    key = find_lang_key(subs, prefer)
    if key:
        return key
    return next(iter(subs), None)


# ── Subtitle Parsing ────────────────────────────────────────────────────────

def parse_json3(data: bytes) -> list[dict]:
    obj = json.loads(data)
    segments = []
    for event in obj.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).strip()
        if not text or text == "\n":
            continue
        segments.append({
            "text": text,
            "start": event.get("tStartMs", 0) / 1000.0,
            "duration": event.get("dDurationMs", 0) / 1000.0,
        })
    return segments


def parse_srv1(data: bytes) -> list[dict]:
    root = ET.fromstring(data)
    segments = []
    for elem in root.iter("text"):
        text = (elem.text or "").strip()
        if not text:
            continue
        start = float(elem.get("start", 0))
        dur = float(elem.get("dur", 0))
        segments.append({"text": text, "start": start, "duration": dur})
    return segments


def parse_vtt(data: bytes) -> list[dict]:
    text = data.decode("utf-8", errors="replace")
    segments = []
    ts_pat = r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})"
    blocks = re.split(r"\n\n+", text)
    for block in blocks:
        m = re.search(ts_pat, block)
        if not m:
            continue
        start = (
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + int(m.group(3))
            + int(m.group(4)) / 1000
        )
        end = (
            int(m.group(5)) * 3600
            + int(m.group(6)) * 60
            + int(m.group(7))
            + int(m.group(8)) / 1000
        )
        lines_after = block[m.end() :].strip()
        line_text = re.sub(r"<[^>]+>", "", lines_after).strip()
        if line_text:
            segments.append({
                "text": line_text,
                "start": round(start, 3),
                "duration": round(end - start, 3),
            })
    return segments


def download_sub_data(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def get_sub_segments(format_list: list[dict]) -> list[dict] | None:
    for preferred_ext in ("json3", "srv1", "vtt"):
        for fmt in format_list:
            if fmt.get("ext") == preferred_ext and fmt.get("url"):
                try:
                    data = download_sub_data(fmt["url"])
                    if preferred_ext == "json3":
                        segs = parse_json3(data)
                    elif preferred_ext == "srv1":
                        segs = parse_srv1(data)
                    else:
                        segs = parse_vtt(data)
                    if segs:
                        return segs
                except Exception as e:
                    log.warning("Failed to parse %s subtitle: %s", preferred_ext, e)
    return None


# ── Output Formatting ───────────────────────────────────────────────────────

def _fmt_ts(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


def _fmt_srt_ts(sec: float) -> str:
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    ms = int((s % 1) * 1000)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"


def segments_to_text(segs: list[dict], timestamps: bool) -> str:
    if timestamps:
        return "\n".join(f"[{_fmt_ts(s['start'])}] {s['text']}" for s in segs)
    return "\n".join(s["text"] for s in segs)


def segments_to_srt(segs: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segs, 1):
        lines.append(str(i))
        lines.append(
            f"{_fmt_srt_ts(s['start'])} --> {_fmt_srt_ts(s['start'] + s['duration'])}"
        )
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


# ── Audio Transcription (OpenRouter Gemini) ──────────────────────────────────

def _gemini_transcribe(audio_b64: str, fmt: str = "mp3") -> str:
    payload = {
        "model": GEMINI_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe this audio accurately. "
                            "Output ONLY the transcription text, nothing else."
                        ),
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": fmt},
                    },
                ],
            }
        ],
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {OPENROUTER_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            result = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        raise RuntimeError(f"OpenRouter HTTP {e.code}: {body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Network error: {e.reason}") from e

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected API response: {json.dumps(result)[:300]}") from e


def _get_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 300.0


def _text_to_segments(text: str, start_offset: float, total_dur: float) -> list[dict]:
    if not text.strip():
        return []
    sentences = re.split(r"(?<=[.!?。！？\n])\s*", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return [{"text": text.strip(), "start": start_offset, "duration": total_dur}]
    seg_dur = total_dur / len(sentences)
    return [
        {
            "text": s,
            "start": round(start_offset + i * seg_dur, 3),
            "duration": round(seg_dur, 3),
        }
        for i, s in enumerate(sentences)
    ]


def _transcribe_audio_file(audio_path: Path) -> list[dict]:
    size_mb = audio_path.stat().st_size / 1024 / 1024

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Compress to mono MP3
        if size_mb > CHUNK_MB_MAX or audio_path.suffix != ".mp3":
            compressed = tmp / "compressed.mp3"
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(audio_path),
                    "-ac", "1", "-ar", "16000", "-b:a", "48k",
                    str(compressed),
                ],
                capture_output=True,
                check=True,
            )
            work_file = compressed
        else:
            work_file = audio_path

        size_mb = work_file.stat().st_size / 1024 / 1024

        if size_mb <= CHUNK_MB_MAX:
            audio_b64 = base64.b64encode(work_file.read_bytes()).decode()
            text = _gemini_transcribe(audio_b64)
            duration = _get_duration(work_file)
            return _text_to_segments(text, 0, duration)

        # Split into chunks
        duration = _get_duration(work_file)
        n_chunks = max(2, int(size_mb / CHUNK_MB_MAX) + 1)
        chunk_sec = max(30, int(duration / n_chunks) + 1)

        chunk_dir = tmp / "chunks"
        chunk_dir.mkdir()
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(work_file),
                "-f", "segment", "-segment_time", str(chunk_sec),
                "-ac", "1", "-ar", "16000", "-b:a", "48k",
                str(chunk_dir / "chunk_%04d.mp3"),
            ],
            capture_output=True,
            check=True,
        )
        chunks = sorted(chunk_dir.glob("chunk_*.mp3"))

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=min(len(chunks), 8)) as pool:
            futures = {}
            for i, chunk in enumerate(chunks):
                b64 = base64.b64encode(chunk.read_bytes()).decode()
                futures[pool.submit(_gemini_transcribe, b64)] = i
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    log.error("Chunk %d transcription failed: %s", idx, e)
                    results[idx] = ""

        all_segments: list[dict] = []
        for i in sorted(results):
            chunk_start = i * chunk_sec
            chunk_dur = min(chunk_sec, duration - chunk_start)
            all_segments.extend(_text_to_segments(results[i], chunk_start, chunk_dur))
        return all_segments


# ── youtube-transcript-api Fallback ──────────────────────────────────────────

def _try_transcript_api(video_id: str, target_lang: str) -> tuple[list[dict] | None, str]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError:
        log.warning("youtube-transcript-api not installed")
        return None, ""

    api = YouTubeTranscriptApi()

    # Build language priority list
    if target_lang in ZH_CODES:
        langs = list(ZH_CODES)
    else:
        langs = [target_lang]

    # Try target language
    try:
        result = api.fetch(video_id, languages=langs)
        segs = [{"text": d["text"], "start": d["start"], "duration": d["duration"]} for d in result.to_raw_data()]
        return segs, target_lang
    except Exception:
        pass

    # Try English
    try:
        result = api.fetch(video_id, languages=["en"])
        segs = [{"text": d["text"], "start": d["start"], "duration": d["duration"]} for d in result.to_raw_data()]
        return segs, "en"
    except Exception:
        pass

    # Try any available
    try:
        transcript_list = api.list(video_id)
        for t in transcript_list:
            try:
                result = t.fetch()
                segs = [
                    {"text": d["text"], "start": d["start"], "duration": d["duration"]}
                    for d in result.to_raw_data()
                ]
                return segs, t.language_code
            except Exception:
                continue
    except Exception:
        pass

    return None, ""


# ── Audio Download + Transcribe ──────────────────────────────────────────────

def _download_and_transcribe(url: str) -> list[dict]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        ydl_opts = {
            "format": "worstaudio/worst",
            "outtmpl": str(tmp / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "48",
                }
            ],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        mp3_files = list(tmp.glob("*.mp3"))
        if not mp3_files:
            audio_files = list(tmp.glob("*.*"))
            if not audio_files:
                raise RuntimeError("No audio file downloaded")
            audio_file = audio_files[0]
        else:
            audio_file = mp3_files[0]

        return _transcribe_audio_file(audio_file)


# ── Main Pipeline ────────────────────────────────────────────────────────────

def fetch_subtitles_pipeline(video_id: str, target_lang: str) -> dict:
    url = f"https://www.youtube.com/watch?v={video_id}"

    # Extract video info
    ydl_opts = {"skip_download": True, "quiet": True, "no_warnings": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise HTTPException(
            404,
            detail={"status": "error", "message": f"Video not found or unavailable: {str(e)[:200]}"},
        )

    if not info:
        raise HTTPException(404, detail={"status": "error", "message": "Video not found"})

    title = info.get("title", "")
    manual_subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    # Priority 1: Manual subtitles (target language)
    key = find_lang_key(manual_subs, target_lang)
    if key:
        segs = get_sub_segments(manual_subs[key])
        if segs:
            log.info("Found manual subs: %s", key)
            return {
                "segments": segs, "language": key,
                "needs_translation": False, "title": title, "status": "ok",
            }

    # Priority 2+3: Auto/translated subtitles (target language)
    key = find_lang_key(auto_subs, target_lang)
    if key:
        segs = get_sub_segments(auto_subs[key])
        if segs:
            log.info("Found auto subs: %s", key)
            return {
                "segments": segs, "language": key,
                "needs_translation": False, "title": title, "status": "ok",
            }

    # Priority 4: Manual subtitles (other language, prefer en)
    key = find_fallback_key(manual_subs, "en")
    if key and not lang_matches(key, target_lang):
        segs = get_sub_segments(manual_subs[key])
        if segs:
            log.info("Found manual subs (fallback): %s", key)
            return {
                "segments": segs, "language": key,
                "needs_translation": True, "title": title, "status": "ok",
            }

    # Priority 5: Auto subtitles (other language, prefer en)
    key = find_fallback_key(auto_subs, "en")
    if key and not lang_matches(key, target_lang):
        segs = get_sub_segments(auto_subs[key])
        if segs:
            log.info("Found auto subs (fallback): %s", key)
            return {
                "segments": segs, "language": key,
                "needs_translation": True, "title": title, "status": "ok",
            }

    # Priority 6: youtube-transcript-api
    try:
        segs, lang = _try_transcript_api(video_id, target_lang)
        if segs:
            needs_trans = not lang_matches(lang, target_lang)
            log.info("Found via transcript API: %s", lang)
            return {
                "segments": segs, "language": lang,
                "needs_translation": needs_trans, "title": title, "status": "ok",
            }
    except Exception as e:
        log.warning("youtube-transcript-api failed: %s", e)

    # Priority 7: Audio download + Gemini transcription
    if not OPENROUTER_KEY:
        raise HTTPException(
            500,
            detail={
                "status": "error",
                "message": "No subtitles available and OPENROUTER_API_KEY not configured",
            },
        )

    log.info("Falling back to audio transcription for %s...", video_id)
    try:
        segs = _download_and_transcribe(url)
        return {
            "segments": segs, "language": "auto",
            "needs_translation": True, "title": title, "status": "transcribed",
        }
    except Exception as e:
        log.error("Audio transcription failed: %s", e)
        raise HTTPException(
            500,
            detail={
                "status": "error",
                "message": f"No subtitles and audio transcription failed: {str(e)[:200]}",
            },
        )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/transcript")
def get_transcript(
    req: TranscriptRequest,
    x_api_key: str | None = Header(default=None),
):
    if not VALID_API_KEY or x_api_key != VALID_API_KEY:
        raise HTTPException(
            401, detail={"status": "error", "message": "Invalid API key"}
        )

    if not req.url:
        raise HTTPException(
            400, detail={"status": "error", "message": "Missing url parameter"}
        )
    if req.format not in ("text", "json", "srt"):
        raise HTTPException(
            400,
            detail={"status": "error", "message": "Invalid format. Use: text, json, srt"},
        )

    try:
        video_id = extract_video_id(req.url)
    except ValueError:
        raise HTTPException(
            400, detail={"status": "error", "message": "Invalid YouTube URL or video ID"}
        )

    result = fetch_subtitles_pipeline(video_id, req.lang)
    segments = result["segments"]

    if req.format == "text":
        transcript = segments_to_text(segments, req.timestamps)
    elif req.format == "srt":
        transcript = segments_to_srt(segments)
    else:
        transcript = ""

    return {
        "status": result["status"],
        "video_id": video_id,
        "title": result["title"],
        "language": result["language"],
        "needs_translation": result["needs_translation"],
        "segment_count": len(segments),
        "segments": segments,
        "transcript": transcript,
    }
