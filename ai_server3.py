"""
ai_server.py - Gemini AI server for the ESP32-S3 + OLED project.

How to run (no terminal needed):
  1. Set the GEMINI_API_KEY environment variable.
  2. In Python IDLE choose  Run > Run Module  (or press F5).
  3. Leave the IDLE Shell window open. The ESP32 sends questions to it.

For a cloud deployment, configure GEMINI_API_KEY in the host's environment
settings. The server uses the host-assigned PORT, or port 5000 when run locally.

The ESP32 sends:   POST http://<laptop-ip>:5000/ask    {"question": "..."}          (typed question)
                   POST http://<laptop-ip>:5000/voice  raw 16-bit 16 kHz mono audio  (microphone)
The server replies: {"oled": "text already wrapped to 21 characters x 8 lines"}

Only Python's built-in libraries are used, so nothing needs to be installed.
(Flask would need a pip install, which is a terminal command, so this uses
Python's built-in web server instead. It does the same job.)
"""

import os
import array
import base64
import io
import json
import re
import socket
import textwrap
import time
import urllib.error
import urllib.request
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------- Settings ----------
SAVED_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MODEL = "gemini-3.5-flash"   # tried first; other Flash models are used automatically if it is busy
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"
PORT = int(os.environ.get("PORT", 5000))
OLED_COLS = 21               # characters per line (6x8 font on 128px)
OLED_ROWS = 8                # lines (8px-tall font on 64px)


# ---------- Text helpers ----------

def clean_key(raw):
    """Remove spaces, quotes and anything not valid in a Google API key."""
    return re.sub(r"[^A-Za-z0-9_.\-]", "", raw.strip())


ASCII_MAP = {
    "\u2192": "->", "\u2190": "<-", "\u2022": "-", "\u2013": "-", "\u2014": "-",
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'", "\u2026": "...",
    "\u00d7": "x", "\u00f7": "/", "\u2248": "~", "\u2264": "<=", "\u2265": ">=",
    "\u00b0": "", "\u2713": "+", "\u2714": "+", "\u00a0": " ",
}


def to_ascii(text):
    """The default OLED font only has ASCII, so swap or drop other characters."""
    for char, ascii_text in ASCII_MAP.items():
        text = text.replace(char, ascii_text)
    return text.encode("ascii", "ignore").decode("ascii")


def strip_markdown(text):
    """Remove markdown formatting but keep numbers, symbols and line breaks."""
    text = re.sub(r"```\w*", "", text)                    # code fences
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # [text](link) -> text
    text = re.sub(r"\*\*|__|`", "", text)                 # bold / code marks
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)      # headings
    text = re.sub(r"(?m)^\s*[*\u2022]\s+", "- ", text)     # bullets -> "- "
    text = re.sub(r"(?m)^>\s?", "", text)                  # quotes
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def wrap_for_oled(text):
    """Wrap each line to 21 characters, keep line breaks, maximum 8 lines."""
    lines = []
    for paragraph in to_ascii(text).split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        is_list_item = re.match(r"^(\d+[.)]|-)\s", paragraph) is not None
        lines.extend(textwrap.wrap(
            paragraph, width=OLED_COLS, break_long_words=True,
            subsequent_indent="   " if is_list_item else "",
        ))
    if len(lines) > OLED_ROWS:
        lines = lines[:OLED_ROWS]
        lines[-1] = lines[-1][: OLED_COLS - 3].rstrip() + "..."
    return "\n".join(lines)


def explain_error(status, message):
    """Turn an API error into a plain-language explanation."""
    if status == 400 and "api key" in message.lower():
        return "Invalid API key. Check that you copied the full key from Google AI Studio."
    if status == 400:
        return "Bad request (400)."
    if status in (401, 403):
        return "Access denied. The key is wrong, disabled, or not allowed to use this model."
    if status == 404:
        return "Model not found. The model name may be wrong or retired. Change MODEL at the top of gemini_web.py."
    if status == 429:
        return "Quota exceeded. You hit the free-tier limit. Wait a minute and retry."
    if status >= 500:
        return "Google server error or overloaded. Try again in a moment."
    return "Unexpected error."


# ---------- Gemini call ----------

class ApiError(RuntimeError):
    """An error with the HTTP status attached (0 = no status, e.g. network down)."""
    def __init__(self, message, status=0):
        super().__init__(message)
        self.status = status


TRANSIENT = {404, 408, 429, 503}   # worth trying the next model
BAD_WORDS = ("tts", "image", "live", "audio", "embedding", "robotics", "computer", "native", "omni", "veo", "lyria")
PREFERRED = {}                     # remembers the model that worked for each key


def get_candidates(api_key):
    """Ask Google which Flash models this key can use, best guesses first."""
    url = API_BASE.rstrip("/") + "?pageSize=200"
    req = urllib.request.Request(url, headers={"x-goog-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            data = json.loads(res.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        try:
            message = json.loads(err.read().decode("utf-8")).get("error", {}).get("message", "")
        except ValueError:
            message = ""
        if err.code in (400, 401, 403):   # the key itself is the problem
            raise ApiError(f"{explain_error(err.code, message)}\nGoogle says: {message}", err.code)
        return [MODEL]
    except (TimeoutError, socket.timeout, urllib.error.URLError) as err:
        raise ApiError(
            "Network error: Python could not reach Google.\n"
            "Check your internet connection, VPN, firewall or antivirus.\n"
            f"Details: {getattr(err, 'reason', err)}"
        )

    names = []
    for m in data.get("models", []):
        name = m.get("name", "").replace("models/", "")
        if ("generateContent" in m.get("supportedGenerationMethods", [])
                and "flash" in name and not any(w in name for w in BAD_WORDS)):
            names.append(name)

    # Order: the model that worked last time, then MODEL, then Flash-Lite, then the rest
    def priority(n):
        if n == PREFERRED.get(api_key): return 0
        if n == MODEL: return 1
        if "preview" in n or "exp" in n: return 4
        return 2 if "lite" in n else 3
    names = sorted(sorted(names, reverse=True), key=priority)
    return names[:5] or [MODEL]


def call_gemini(api_key, model, prompt, system_text, max_tokens, audio_b64=None):
    """One request to one model. Returns the reply text, or raises ApiError."""
    url = f"{API_BASE}{model}:generateContent"

    def build_request(with_thinking):
        config = {"maxOutputTokens": max_tokens}
        if with_thinking:
            config["thinkingConfig"] = {"thinkingLevel": "low"}  # less "thinking" = faster
        parts = [{"text": prompt}]
        if audio_b64:   # spoken question: attach the recording before the text
            parts.insert(0, {"inlineData": {"mimeType": "audio/wav", "data": audio_b64}})
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system_text}]},
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": config,
        }).encode("utf-8")
        return urllib.request.Request(
            url, data=body, method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
        )

    use_thinking = True
    status, data = 0, {}
    for _ in range(2):                 # second try only if the thinking setting is rejected
        started = time.time()
        try:
            with urllib.request.urlopen(build_request(use_thinking), timeout=45) as res:
                status, data = res.status, json.loads(res.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            status = err.code
            try:
                data = json.loads(err.read().decode("utf-8"))
            except ValueError:
                data = {}
        except (TimeoutError, socket.timeout):
            raise ApiError("Timed out: Google did not answer within 45 seconds.", 408)
        except urllib.error.URLError as err:
            raise ApiError(
                "Network error: Python could not reach Google.\n"
                "Check your internet connection, VPN, firewall or antivirus.\n"
                f"Details: {err.reason}"
            )
        print(f"  {model}: status {status} in {time.time() - started:.1f}s")

        message = data.get("error", {}).get("message", "")
        if status == 400 and use_thinking and "thinking" in message.lower():
            use_thinking = False       # this model does not accept the thinking setting
            continue
        break

    # Google sends errors as { "error": { "code", "message", "status" } }
    if status != 200:
        message = data.get("error", {}).get("message", f"HTTP {status}")
        raise ApiError(f"{explain_error(status, message)}\nGoogle says: {message}", status)

    # candidates and parts are lists -> take the first of each
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError, TypeError):
        reason = (
            data.get("promptFeedback", {}).get("blockReason")
            or (data.get("candidates") or [{}])[0].get("finishReason")
            or "unknown"
        )
        raise ApiError(f"Gemini returned no text (reason: {reason}). The prompt may have been blocked.")


OLED_SYSTEM = (
    "Answer in exactly 2 short sentences, under 150 characters in total. "
    "Plain text only: no markdown, no asterisks, no lists, no emojis, no special symbols."
)


VOICE_SYSTEM = (
    OLED_SYSTEM +
    " The question is spoken in the attached audio: answer it. "
    "If the audio is silent or unclear, reply exactly: Sorry, I did not catch that."
)
VOICE_PROMPT = "Answer the spoken question in the audio."


def pcm_to_wav_base64(pcm, rate):
    """Wrap raw 16-bit mono audio in a WAV file and base64-encode it for Gemini.
    The audio is also made louder if the microphone was very quiet."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) // 2 * 2])
    peak = max((abs(v) for v in samples), default=0)
    print(f"  Audio: {len(samples) / rate:.1f} s, peak level {peak}")
    if 0 < peak < 20000:
        gain = min(20000 / peak, 20)
        samples = array.array("h", (int(v * gain) for v in samples))
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(samples.tobytes())
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def answer(api_key, question, audio_b64=None, system_text=None):
    """ONE request returns the AI-style answer for the OLED.
    If a model is busy or over its limit, the next available model is tried."""
    candidates = get_candidates(api_key)
    print("  Models to try:", ", ".join(candidates))

    text, last = None, None
    for model in candidates:
        try:
            text = call_gemini(api_key, model, question, system_text or OLED_SYSTEM, 1024, audio_b64)
            PREFERRED[api_key] = model
            break
        except ApiError as err:
            last = err
            if err.status not in TRANSIENT:
                raise
            print(f"  {model} failed ({err.status}), trying the next model...")
    if text is None:
        raise ApiError(
            "Every available model is busy or over its free limit "
            f"(tried: {', '.join(candidates)}).\n"
            "Free-tier limits are per minute and per day. Wait a minute and try again.\n"
            f"Last error: {last}"
        )
    return {"oled": wrap_for_oled(strip_markdown(text))}


# ---------- Local server for the ESP32 ----------

class Handler(BaseHTTPRequestHandler):
    def _send(self, code, content_type, payload):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        # Open http://<laptop-ip>:5000 in a browser to test that the server is reachable
        text = f"AI server is running. Model: {MODEL}. The ESP32 should POST to /ask or /voice."
        self._send(200, "text/plain; charset=utf-8", text.encode("utf-8"))

    def do_POST(self):
        if self.path not in ("/ask", "/voice"):
            return self._send(404, "text/plain", b"Not found")
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 2_000_000:
                raise RuntimeError("Recording too large.")
            raw = self.rfile.read(length)
            api_key = clean_key(SAVED_API_KEY)
            if not api_key:
                raise RuntimeError("No API key. Set the GEMINI_API_KEY environment variable.")

            if self.path == "/ask":                       # typed question
                question = str(json.loads(raw.decode("utf-8")).get("question", "")).strip()
                if not question:
                    raise RuntimeError("Empty question.")
                print(f"Question: {question[:80]}")
                result = answer(api_key, question)
            else:                                         # spoken question
                rate = int(self.headers.get("X-Sample-Rate", "16000"))
                if len(raw) < rate:                       # less than 0.5 seconds of audio
                    raise RuntimeError("Recording too short.")
                print("Voice question received")
                result = answer(api_key, VOICE_PROMPT, pcm_to_wav_base64(raw, rate), VOICE_SYSTEM)
            print("  Answer sent to the ESP32.")
        except (RuntimeError, ApiError) as err:
            print(f"  Error: {err}")
            first_line = str(err).split("\n")[0]
            result = {"error": str(err), "oled": wrap_for_oled("Error: " + first_line)}
        except Exception as err:  # anything unexpected
            print(f"  Unexpected error: {err}")
            result = {"error": str(err), "oled": wrap_for_oled("Server error")}
        self._send(200, "application/json", json.dumps(result).encode("utf-8"))

    def log_message(self, *args):
        pass  # keep the Shell window quiet


def laptop_ip_guess():
    """Best guess of this laptop's Wi-Fi IP (double-check it in Windows Settings)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))   # no data is sent; this only picks the active network
            return s.getsockname()[0]
    except OSError:
        return "unknown"


def main():
    if not clean_key(SAVED_API_KEY):
        print("WARNING: GEMINI_API_KEY is empty. Set it in your environment or cloud service settings.\n")
    try:
        server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)   # 0.0.0.0 = reachable from the ESP32
    except OSError:
        print(f"Port {PORT} is already in use. Another copy of this script is probably running.")
        print("Close other IDLE windows running ai_server.py (or use Shell > Restart Shell), then run again.")
        return
    print(f"AI server running on port {PORT}.")
    print(f"Your laptop IP is probably: {laptop_ip_guess()}  (confirm in Windows Settings > Wi-Fi > Properties)")
    print("Put that IP in LAPTOP_IP in the Arduino code. Keep this window open.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


if __name__ == "__main__":
    main()
