"""
JagX extensions — free news, maps/geocode, idle self-learn.
Loaded after core app. Created by JagX & JRILICENSE.
All services used here are free (no Google Maps billing).
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlencode

import requests

logger = logging.getLogger("jagx-ai")

try:
    HTTP  # type: ignore  # noqa: F821
except NameError:
    HTTP = requests.Session()

NEWS_FEEDS = [
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Africa", "https://feeds.bbci.co.uk/news/world/africa/rss.xml"),
    ("Reuters World", "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Premium Times NG", "https://www.premiumtimesng.com/feed"),
    ("Punch NG", "https://punchng.com/feed/"),
]

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
OSM_UA = os.environ.get("JAGX_OSM_USER_AGENT", "JagXAI/7.0 (education; contact: jagx)")

SELF_LEARN_INTERVAL = int(os.environ.get("JAGX_SELF_LEARN_SECONDS", "1800"))
SELF_LEARN_FILE = "knowledge_pack_self_learned.json"
TRAINING_DATA_FILE = globals().get("TRAINING_DATA_FILE", "jagx_training_data.jsonl")
MAX_SELF_ENTRIES = 200

_self_learn_lock = threading.Lock()
_last_self_learn = 0.0


def fetch_news(topic: str = "", max_items: int = 8) -> str:
    topic_l = (topic or "").lower().strip()
    collected = []
    for name, url in NEWS_FEEDS:
        try:
            r = HTTP.get(url, timeout=12, headers={"User-Agent": OSM_UA})
            if r.status_code != 200:
                continue
            root = ET.fromstring(r.content)
            items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
            for it in items[:12]:
                title_el = it.find("title")
                if title_el is None:
                    title_el = it.find("{http://www.w3.org/2005/Atom}title")
                desc_el = it.find("description")
                if desc_el is None:
                    desc_el = it.find("{http://www.w3.org/2005/Atom}summary")
                title = (title_el.text or "").strip() if title_el is not None else ""
                desc = (desc_el.text or "").strip() if desc_el is not None else ""
                desc = re.sub(r"<[^>]+>", "", desc)[:220]
                if not title:
                    continue
                blob = f"{title} {desc}".lower()
                if topic_l and topic_l not in blob and not any(w in blob for w in topic_l.split() if len(w) > 3):
                    continue
                collected.append(f"**{title}** ({name})\n{desc}" if desc else f"**{title}** ({name})")
                if len(collected) >= max_items:
                    break
        except Exception as e:
            logger.warning("news feed %s: %s", name, e)
        if len(collected) >= max_items:
            break
    if not collected:
        return "No fresh RSS headlines matched. Try: search the web for the topic, or ask for world / Africa / Nigeria news."
    header = f"Latest news{(' about ' + topic) if topic else ''} (free RSS feeds):\n\n"
    return header + "\n\n".join(collected)


def geocode_place(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Please give a place name, address, or area to look up."
    try:
        params = {"q": q, "format": "json", "limit": 3, "addressdetails": 1}
        r = HTTP.get(
            f"{NOMINATIM_URL}/search",
            params=params,
            timeout=15,
            headers={"User-Agent": OSM_UA},
        )
        if r.status_code != 200:
            return "Map lookup is busy right now. Try again shortly."
        data = r.json()
        if not data:
            return f"No place found for “{q}”. Try a clearer name or city."
        lines = []
        for row in data:
            lat = row.get("lat")
            lon = row.get("lon")
            disp = row.get("display_name", "")
            lines.append(f"- **{disp}**\n  Coordinates: {lat}, {lon}\n  Map: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=14/{lat}/{lon}")
        return "Places found (OpenStreetMap, free):\n\n" + "\n\n".join(lines) + "\n\nPrivacy: JagX does not track your live GPS unless you type a place or send coordinates yourself."
    except Exception as e:
        logger.warning("geocode: %s", e)
        return "Map service error. Try again or search the web for the place."


def reverse_geocode(lat: float, lon: float) -> str:
    try:
        params = {"lat": lat, "lon": lon, "format": "json"}
        r = HTTP.get(
            f"{NOMINATIM_URL}/reverse",
            params=params,
            timeout=15,
            headers={"User-Agent": OSM_UA},
        )
        if r.status_code != 200:
            return "Could not reverse-lookup those coordinates."
        data = r.json()
        name = data.get("display_name") or "Unknown place"
        return f"**Near:** {name}\nCoordinates: {lat}, {lon}\nMap: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=15/{lat}/{lon}"
    except Exception as e:
        return f"Reverse geocode failed: {e}"


def weather_for_place(place: str) -> str:
    q = (place or "").strip()
    if not q:
        return "Name a city or area for the weather."
    try:
        params = {"q": q, "format": "json", "limit": 1}
        r = HTTP.get(f"{NOMINATIM_URL}/search", params=params, timeout=12, headers={"User-Agent": OSM_UA})
        data = r.json() if r.status_code == 200 else []
        if not data:
            return f"Could not find “{q}” on the map."
        lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
        name = data[0].get("display_name", q)
        wr = HTTP.get(
            "https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon, "current_weather": True},
            timeout=12,
        )
        if wr.status_code != 200:
            return f"Found {name} but weather feed is unavailable."
        cw = wr.json().get("current_weather") or {}
        return (
            f"**Weather near {name}**\n"
            f"Temperature: {cw.get('temperature')} °C\n"
            f"Wind: {cw.get('windspeed')} km/h\n"
            f"Time (UTC-related): {cw.get('time')}\n"
            f"(Open-Meteo + OpenStreetMap — free, no Google key)"
        )
    except Exception as e:
        return f"Weather lookup failed: {e}"


def run_self_learn_once() -> dict:
    global _last_self_learn
    with _self_learn_lock:
        now = time.time()
        if now - _last_self_learn < 60:
            return {"ok": False, "reason": "cooldown"}
        _last_self_learn = now

        entries = []
        path = TRAINING_DATA_FILE
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()[-300:]
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    q = (row.get("input") or row.get("q") or "").strip()
                    a = (row.get("output") or row.get("a") or "").strip()
                    if len(q) < 8 or len(a) < 20 or len(q) > 180:
                        continue
                    if a.startswith("{") or "traceback" in a.lower():
                        continue
                    keys = [q.lower()[:120]]
                    words = re.findall(r"[a-z0-9]{4,}", q.lower())
                    if len(words) >= 2:
                        keys.append(" ".join(words[:6]))
                    entries.append({"k": keys, "a": a[:1500]})
            except Exception as e:
                logger.warning("self_learn read: %s", e)

        seen = set()
        unique = []
        for e in entries:
            k0 = e["k"][0]
            if k0 in seen:
                continue
            seen.add(k0)
            unique.append(e)
        unique = unique[-MAX_SELF_ENTRIES:]

        try:
            with open(SELF_LEARN_FILE, "w", encoding="utf-8") as f:
                json.dump(unique, f, ensure_ascii=False, indent=2)
        except Exception as e:
            return {"ok": False, "reason": str(e)}

        try:
            if "load_local_brain" in globals() and callable(globals()["load_local_brain"]):
                globals()["load_local_brain"]()
            elif "LOCAL_KB" in globals():
                kb = globals()["LOCAL_KB"]
                for e in unique:
                    keys = e.get("k") or []
                    ans = e.get("a") or ""
                    if keys and ans:
                        kb.append({"keys": [str(k).lower() for k in keys], "answer": ans})
        except Exception as e:
            logger.warning("self_learn reload: %s", e)

        return {"ok": True, "entries": len(unique), "file": SELF_LEARN_FILE}


def _idle_self_learn_loop():
    while True:
        try:
            time.sleep(max(300, SELF_LEARN_INTERVAL))
            result = run_self_learn_once()
            logger.info("idle self-learn: %s", result)
        except Exception as e:
            logger.warning("idle self-learn loop: %s", e)
            time.sleep(300)


def register_extension_routes():
    app = globals().get("app")
    if app is None:
        logger.warning("extensions: no app object")
        return

    @app.get("/news")
    def news_endpoint(topic: str = ""):
        return {"news": fetch_news(topic)}

    @app.get("/geo")
    def geo_endpoint(q: str = ""):
        return {"result": geocode_place(q)}

    @app.get("/weather")
    def weather_endpoint(place: str = ""):
        return {"result": weather_for_place(place)}

    @app.post("/self_learn")
    def self_learn_endpoint():
        return run_self_learn_once()

    @app.get("/extensions")
    def extensions_info():
        return {
            "news": "GET /news?topic=nigeria",
            "maps": "GET /geo?q=Lagos (OpenStreetMap — free, not Google)",
            "weather": "GET /weather?place=Abuja (Open-Meteo free)",
            "self_learn": "POST /self_learn — turns chat logs into local knowledge when idle",
            "privacy": "No silent GPS tracking. User must provide place name or coordinates.",
        }

    logger.info("JagX extensions registered: news, geo, weather, self_learn")


def tool_dispatch_extra(tool_name: str, tool_input) -> Optional[str]:
    name = (tool_name or "").lower().strip()
    if name in ("news", "fetch_news", "latest_news"):
        topic = tool_input if isinstance(tool_input, str) else (tool_input or {}).get("topic", "")
        return fetch_news(str(topic or ""))
    if name in ("geocode", "maps", "location", "where_is"):
        q = tool_input if isinstance(tool_input, str) else (tool_input or {}).get("query", "")
        return geocode_place(str(q or ""))
    if name in ("reverse_geocode",):
        if isinstance(tool_input, dict):
            return reverse_geocode(float(tool_input.get("lat")), float(tool_input.get("lon")))
        return "Need lat and lon."
    if name in ("weather",):
        place = tool_input if isinstance(tool_input, str) else (tool_input or {}).get("place", "")
        return weather_for_place(str(place or ""))
    return None


try:
    register_extension_routes()
except Exception as e:
    logger.warning("register_extension_routes: %s", e)

if os.environ.get("JAGX_SELF_LEARN_ENABLED", "true").lower() == "true":
    t = threading.Thread(target=_idle_self_learn_loop, name="jagx-self-learn", daemon=True)
    t.start()
    logger.info("Idle self-learn thread started (interval=%ss)", SELF_LEARN_INTERVAL)
