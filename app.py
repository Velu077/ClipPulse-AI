import os
import sys
import glob
import shutil
import json
import uuid
import subprocess
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
import whisper

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_SCRIPTS_DIR = os.path.dirname(sys.executable)

# --- CROSS-PLATFORM FFMPEG SETUP (Windows + Linux Cloud) ---
def setup_ffmpeg():
    # On Linux Cloud (GitHub Actions / Docker / Hugging Face), use system ffmpeg directly
    if os.name != "nt":
        return "ffmpeg"

    local_ffmpeg = os.path.join(PROJECT_DIR, "ffmpeg.exe")
    if not os.path.exists(local_ffmpeg):
        winget_pattern = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\**\ffmpeg.exe")
        matches = glob.glob(winget_pattern, recursive=True)
        if matches:
            shutil.copy2(matches[0], local_ffmpeg)
        else:
            try:
                import imageio_ffmpeg
                shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), local_ffmpeg)
            except Exception as e:
                print(f"FFmpeg auto-copy warning: {e}")

    os.environ["PATH"] = (
        PROJECT_DIR + os.pathsep +
        VENV_SCRIPTS_DIR + os.pathsep +
        os.environ.get("PATH", "")
    )
    return local_ffmpeg if os.path.exists(local_ffmpeg) else "ffmpeg"

FFMPEG_EXE = setup_ffmpeg() if os.name == "nt" else "ffmpeg"

DOWNLOADS_DIR = os.path.join(PROJECT_DIR, "downloads")
OUTPUTS_DIR = os.path.join(PROJECT_DIR, "outputs")
STATE_FILE = os.path.join(PROJECT_DIR, "agent_state.json")

os.makedirs(DOWNLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = FastAPI(title="ClipPulse AI Autonomous Agent")
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

def load_state():
    default_state = {
        "agent_enabled": True,
        "check_interval_minutes": 15,
        "gemini_api_key": os.environ.get("GEMINI_API_KEY", ""),
        "default_layout": "blur",
        "burn_subtitles": True,
        "watched_channels": [],
        "processed_video_ids": [],
        "jobs": [],
        "clips": []
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                loaded = json.load(f)
                default_state.update(loaded)
        except Exception:
            pass
    return default_state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

state = load_state()
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("base")
    return whisper_model

def log_job(job_id, status, message):
    for job in state["jobs"]:
        if job["id"] == job_id:
            job["status"] = status
            job["message"] = message
            job["updated_at"] = datetime.now().strftime("%H:%M:%S")
            break
    save_state(state)

# --- SRT SUBTITLE GENERATOR ---
def format_srt_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def generate_clip_srt(segments, clip_start: float, clip_end: float, srt_path: str) -> bool:
    idx = 1
    with open(srt_path, "w", encoding="utf-8") as f:
        for s in segments:
            if s["end"] > clip_start and s["start"] < clip_end:
                st = max(0.0, s["start"] - clip_start)
                et = max(st + 0.4, min(clip_end - clip_start, s["end"] - clip_start))
                text = s["text"].strip()
                if text:
                    f.write(f"{idx}\n{format_srt_time(st)} --> {format_srt_time(et)}\n{text}\n\n")
                    idx += 1
    return idx > 1

# --- CORE AI VIDEO CLIPPING PIPELINE ---
def process_video_pipeline(
    youtube_url: str,
    job_id: str,
    source: str = "Manual",
    layout_mode: str = "blur",
    burn_subs: bool = True
):
    try:
        log_job(job_id, "Downloading", f"Fetching video ({layout_mode.upper()} mode)...")
        raw_video = os.path.join(DOWNLOADS_DIR, f"{job_id}.mp4")
        if os.path.exists(raw_video):
            os.remove(raw_video)

        cmd = [
            sys.executable, "-m", "yt_dlp",
            "--ffmpeg-location", FFMPEG_EXE,
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
            "--merge-output-format", "mp4",
            "-o", raw_video,
            youtube_url
        ]
        subprocess.run(cmd, check=True)

        log_job(job_id, "Transcribing", "Running Whisper Speech-to-Text...")
        model = get_whisper_model()
        result = model.transcribe(raw_video, fp16=False)
        segments = [
            {"start": round(s["start"], 1), "end": round(s["end"], 1), "text": s["text"].strip()}
            for s in result.get("segments", [])
        ]

        highlights = []
        api_key = state.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
        if api_key and segments:
            try:
                import google.generativeai as genai
                log_job(job_id, "AI Analyzing", "Gemini selecting complete viral hooks...")
                genai.configure(api_key=api_key)
                llm = genai.GenerativeModel("gemini-1.5-pro")
                prompt = (
                    "Analyze these timestamped transcript segments and return ONLY a raw JSON array of the top 2 "
                    "most engaging 30-to-45 second hooks. Ensure start and end align with complete sentences. Format: "
                    '[{"start": 10.0, "end": 42.0, "title": "Hook Title", "score": 94}]\n'
                    f"Segments: {json.dumps(segments[:80])}"
                )
                resp = llm.generate_content(prompt)
                clean_json = resp.text.strip().removeprefix("```json").removesuffix("```").strip()
                highlights = json.loads(clean_json)
            except Exception as e:
                print(f"Gemini fallback triggered: {e}")

        if not highlights:
            if segments:
                start_candidates = [s["start"] for s in segments if s["start"] >= 2.0]
                clip_start = start_candidates[0] if start_candidates else segments[0]["start"]
                target_end = clip_start + 30.0
                end_candidates = [s["end"] for s in segments if s["end"] >= target_end]
                clip_end = end_candidates[0] if end_candidates else segments[-1]["end"]
                if clip_end - clip_start > 45.0:
                    clip_end = clip_start + 35.0
            else:
                clip_start, clip_end = 0.0, 30.0

            highlights = [{
                "start": clip_start,
                "end": clip_end,
                "title": f"Smart Hook ({layout_mode.capitalize()})",
                "score": 92
            }]

        log_job(job_id, "Rendering 9:16", f"Rendering {len(highlights)} vertical short(s)...")
        for idx, h in enumerate(highlights):
            out_filename = f"clip_{job_id}_{idx+1}.mp4"
            out_path = os.path.join(OUTPUTS_DIR, out_filename)
            srt_filename = f"subs_{job_id}_{idx+1}.srt"
            srt_path = os.path.join(PROJECT_DIR, srt_filename)

            start_t = float(h["start"])
            end_t = float(h["end"])
            duration = max(5.0, end_t - start_t)

            has_subs = False
            if burn_subs and segments:
                has_subs = generate_clip_srt(segments, start_t, end_t, srt_path)

            sub_style = (
                f",subtitles={srt_filename}:force_style="
                "'Alignment=2,MarginV=65,Fontsize=15,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2,Bold=1'"
                if has_subs else ""
            )

            if layout_mode == "blur":
                vf_filter = (
                    "split[bg][fg];"
                    "[bg]crop=ih*(9/16):ih:(iw-ow)/2:0,scale=1080:1920,boxblur=25:5[bg_blur];"
                    "[fg]scale=1080:-2[fg_scaled];"
                    f"[bg_blur][fg_scaled]overlay=(W-w)/2:(H-h)/2{sub_style}"
                )
            else:
                vf_filter = (
                    "crop=iw:ih*0.80:0:ih*0.10,"
                    "crop=ih*(9/16):ih:(iw-ow)/2:0,"
                    f"scale=1080:1920{sub_style}"
                )

            ffmpeg_cmd = [
                FFMPEG_EXE, "-y",
                "-ss", str(start_t),
                "-t", str(duration),
                "-i", raw_video,
                "-vf", vf_filter,
                "-c:v", "libx264", "-crf", "22", "-preset", "fast",
                "-c:a", "aac",
                out_path
            ]
            subprocess.run(ffmpeg_cmd, cwd=PROJECT_DIR, check=True)
            if os.path.exists(srt_path):
                os.remove(srt_path)

            state["clips"].insert(0, {
                "filename": out_filename,
                "url": f"/outputs/{out_filename}",
                "title": h.get("title", f"Short #{idx+1}"),
                "score": h.get("score", 92),
                "layout": layout_mode.upper(),
                "source_url": youtube_url,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M")
            })

        log_job(job_id, "Completed", f"Rendered {len(highlights)} short(s) in {layout_mode.upper()} mode ({source})")
    except Exception as e:
        log_job(job_id, "Failed", str(e)[:120])

# --- AUTONOMOUS AGENT BACKGROUND LOOP ---
def autonomous_agent_tick():
    if not state.get("agent_enabled") or not state.get("watched_channels"):
        return
    for channel_url in state["watched_channels"]:
        try:
            cmd = [sys.executable, "-m", "yt_dlp", "--playlist-end", "1", "--get-id", channel_url]
            res = subprocess.run(cmd, capture_output=True, text=True, check=True)
            video_id = res.stdout.strip().splitlines()[0]
            if video_id and video_id not in state["processed_video_ids"]:
                state["processed_video_ids"].append(video_id)
                video_url = f"https://www.youtube.com/watch?v={video_id}"
                job_id = str(uuid.uuid4())[:8]
                state["jobs"].insert(0, {
                    "id": job_id,
                    "url": video_url,
                    "status": "Queued (Agent)",
                    "message": "Auto-detected new upload",
                    "updated_at": datetime.now().strftime("%H:%M:%S")
                })
                save_state(state)
                process_video_pipeline(
                    video_url,
                    job_id,
                    source="Autonomous Agent",
                    layout_mode=state.get("default_layout", "blur"),
                    burn_subs=state.get("burn_subtitles", True)
                )
        except Exception as e:
            print(f"[Autonomous Agent] Error checking {channel_url}: {e}")

scheduler = BackgroundScheduler()
scheduler.add_job(autonomous_agent_tick, "interval", minutes=15, id="agent_loop")
scheduler.start()

class ClipRequest(BaseModel):
    url: str
    layout_mode: str = "blur"
    burn_subtitles: bool = True

class AgentConfigRequest(BaseModel):
    channel_url: str = ""
    gemini_api_key: str = ""
    default_layout: str = "blur"
    burn_subtitles: bool = True
    agent_enabled: bool = True

@app.get("/api/state")
def get_state():
    return state

@app.post("/api/clip")
def trigger_clip(req: ClipRequest, bg: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    state["default_layout"] = req.layout_mode
    state["burn_subtitles"] = req.burn_subtitles
    state["jobs"].insert(0, {
        "id": job_id,
        "url": req.url,
        "status": "Queued",
        "message": f"Starting ({req.layout_mode.upper()} mode)...",
        "updated_at": datetime.now().strftime("%H:%M:%S")
    })
    save_state(state)
    bg.add_task(process_video_pipeline, req.url, job_id, "Web UI", req.layout_mode, req.burn_subtitles)
    return {"status": "started", "job_id": job_id}

@app.post("/api/agent")
def update_agent(cfg: AgentConfigRequest, bg: BackgroundTasks):
    state["agent_enabled"] = cfg.agent_enabled
    state["default_layout"] = cfg.default_layout
    state["burn_subtitles"] = cfg.burn_subtitles
    if cfg.gemini_api_key.strip():
        state["gemini_api_key"] = cfg.gemini_api_key.strip()
    if cfg.channel_url.strip() and cfg.channel_url.strip() not in state["watched_channels"]:
        state["watched_channels"].append(cfg.channel_url.strip())
    save_state(state)
    bg.add_task(autonomous_agent_tick)
    return {"status": "updated", "state": state}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ClipPulse AI — Studio & Autonomous Agent</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen font-sans p-4 md:p-8">
    <div class="max-w-6xl mx-auto space-y-6">
        <header class="flex flex-wrap items-center justify-between border-b border-slate-800 pb-4 gap-4">
            <div>
                <h1 class="text-2xl font-bold bg-gradient-to-r from-indigo-400 to-cyan-400 bg-clip-text text-transparent">
                    ClipPulse AI Studio & Autonomous Agent
                </h1>
                <p class="text-xs text-slate-400">Smart Sentence-Snapping • 9:16 Blur Canvas & Face-Zoom • Burned-in Subtitles</p>
            </div>
            <span class="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                ● Autonomous Agent Active
            </span>
        </header>

        <div class="grid md:grid-cols-2 gap-6">
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
                <h2 class="font-semibold text-lg text-indigo-300">1. Instant Video Clipper</h2>
                <input id="videoUrl" type="text" placeholder="Paste YouTube URL (https://youtu.be/...)" 
                    class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500">
                
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <button id="btnBlur" onclick="setMode('blur')" class="p-2.5 rounded-lg border border-indigo-500 bg-indigo-500/20 text-indigo-200 font-semibold text-left">
                        🎬 Blur Canvas (Trailers/Movies)<br>
                        <span class="text-[10px] text-slate-400 font-normal">Keeps full 16:9 text & wide shots</span>
                    </button>
                    <button id="btnCrop" onclick="setMode('crop')" class="p-2.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 font-semibold text-left">
                        🎙️ 9:16 Face-Zoom (Podcasts)<br>
                        <span class="text-[10px] text-slate-500 font-normal">Strips black bars & fills vertical</span>
                    </button>
                </div>

                <div class="flex items-center justify-between pt-1">
                    <label class="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                        <input id="burnSubs" type="checkbox" checked class="accent-indigo-500 w-4 h-4">
                        Burn Dynamic Yellow Subtitles into Video
                    </label>
                    <button onclick="submitClip()" class="bg-indigo-600 hover:bg-indigo-500 px-5 py-2 rounded-lg text-sm font-semibold">
                        Generate Shorts
                    </button>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
                <h2 class="font-semibold text-lg text-cyan-300">2. Hands-Free Autonomous Agent</h2>
                <p class="text-xs text-slate-400">Checks watched channels every 15 mins and auto-renders using your selected layout mode.</p>
                <input id="channelUrl" type="text" placeholder="YouTube Channel / Playlist URL to auto-watch..." 
                    class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                <div class="flex gap-2">
                    <input id="geminiKey" type="password" placeholder="Optional: Gemini API Key for AI Hooks" 
                        class="w-full bg-slate-950 border border-slate-700 rounded-lg px-3 py-2 text-sm">
                    <button onclick="saveAgent()" class="bg-cyan-600 hover:bg-cyan-500 px-4 py-2 rounded-lg text-sm font-semibold whitespace-nowrap">
                        Save & Run Agent
                    </button>
                </div>
                <div id="watchedList" class="text-xs text-slate-400 pt-1"></div>
            </div>
        </div>

        <div class="bg-slate-900 border border-slate-800 rounded-xl p-5">
            <h2 class="font-semibold text-sm text-slate-300 mb-3 uppercase tracking-wider">Live Background Jobs</h2>
            <div id="jobsList" class="space-y-2 text-sm text-slate-400">Loading...</div>
        </div>

        <div>
            <h2 class="font-semibold text-lg mb-4">Generated 9:16 Vertical Shorts</h2>
            <div id="clipsGrid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6"></div>
        </div>
    </div>

    <script>
        let currentMode = 'blur';

        function setMode(mode) {
            currentMode = mode;
            const bBlur = document.getElementById('btnBlur');
            const bCrop = document.getElementById('btnCrop');
            if (mode === 'blur') {
                bBlur.className = "p-2.5 rounded-lg border border-indigo-500 bg-indigo-500/20 text-indigo-200 font-semibold text-left";
                bCrop.className = "p-2.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 font-semibold text-left";
            } else {
                bCrop.className = "p-2.5 rounded-lg border border-indigo-500 bg-indigo-500/20 text-indigo-200 font-semibold text-left";
                bBlur.className = "p-2.5 rounded-lg border border-slate-800 bg-slate-950 text-slate-400 font-semibold text-left";
            }
        }

        async function refreshState() {
            const res = await fetch('/api/state');
            const data = await res.json();
            
            document.getElementById('watchedList').innerHTML = data.watched_channels.length 
                ? 'Watching: ' + data.watched_channels.join(', ') 
                : 'No channels watched yet.';

            const jobsHtml = data.jobs.slice(0, 5).map(j => `
                <div class="flex items-center justify-between bg-slate-950 px-3 py-2 rounded border border-slate-800">
                    <span class="truncate max-w-md">${j.url}</span>
                    <span class="text-xs px-2 py-1 rounded bg-indigo-500/20 text-indigo-300">${j.status}: ${j.message}</span>
                </div>
            `).join('') || '<p class="text-xs text-slate-500">No active jobs.</p>';
            document.getElementById('jobsList').innerHTML = jobsHtml;

            const clipsHtml = data.clips.map(c => `
                <div class="bg-slate-900 border border-slate-800 rounded-xl overflow-hidden flex flex-col">
                    <video src="${c.url}" controls class="w-full aspect-[9/16] bg-black object-cover"></video>
                    <div class="p-3 flex items-center justify-between">
                        <div>
                            <p class="font-semibold text-sm">${c.title}</p>
                            <p class="text-xs text-emerald-400">Score: ${c.score} • ${c.layout || '9:16'}</p>
                        </div>
                        <a href="${c.url}" download class="bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded text-xs font-medium">Download</a>
                    </div>
                </div>
            `).join('') || '<p class="text-sm text-slate-500 col-span-3">No clips generated yet. Paste a URL above!</p>';
            document.getElementById('clipsGrid').innerHTML = clipsHtml;
        }

        async function submitClip() {
            const url = document.getElementById('videoUrl').value.trim();
            if (!url) return;
            const burn_subtitles = document.getElementById('burnSubs').checked;
            document.getElementById('videoUrl').value = '';
            await fetch('/api/clip', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({url, layout_mode: currentMode, burn_subtitles})
            });
            refreshState();
        }

        async function saveAgent() {
            const channel_url = document.getElementById('channelUrl').value.trim();
            const gemini_api_key = document.getElementById('geminiKey').value.trim();
            const burn_subtitles = document.getElementById('burnSubs').checked;
            await fetch('/api/agent', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    channel_url,
                    gemini_api_key,
                    default_layout: currentMode,
                    burn_subtitles,
                    agent_enabled: true
                })
            });
            document.getElementById('channelUrl').value = '';
            refreshState();
        }

        refreshState();
        setInterval(refreshState, 4000);
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860 if os.name != "nt" else 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
