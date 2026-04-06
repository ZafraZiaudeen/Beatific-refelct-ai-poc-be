# Reflect AI POC Backend

This backend runs the same-device voice session, summary generation, and speaker identification bridge.

## Services

- Node API and WebSocket server on `http://localhost:3100`
- Python speaker ID service on `http://127.0.0.1:8200`

## Local Run

1. Copy `.env.example` to `.env`.
2. Install Node dependencies with `npm install`.
3. Install Python dependencies inside `python-service` with `pip install -r requirements.txt`.
4. If you already created the venv before this change, refresh the Python packages:

```powershell
cd python-service
pip install -r requirements.txt --upgrade
```

Git Bash equivalent:

```bash
cd python-service
./venv/Scripts/python.exe -m pip install -r requirements.txt --upgrade
```

5. Start the Python speaker ID service:

```powershell
cd python-service
uvicorn app:app --host 127.0.0.1 --port 8200
```

Git Bash equivalent:

```bash
cd python-service
./venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8200
```

6. Start the Node backend:

```powershell
npm run dev
```

The frontend in `../reflect-ai-poc` should point to `http://localhost:3100`.

If `speechbrain` still cannot import cleanly, the service now falls back to a lighter local acoustic embedding backend so the app can still boot for testing.

Notes:

- The PowerShell commands use Windows-style paths like `venv\Scripts\python`. In Git Bash, use `./venv/Scripts/python.exe` instead.
- If Git Bash keeps printing `bash: sed: command not found` or `bash: uname: command not found`, your Git Bash environment/path is misconfigured. That is separate from the Python app itself.
- For the full SpeechBrain ECAPA path, keep `huggingface_hub` below `1.0`. The current app will still run with a fallback embedding backend if that stack is mismatched or offline.
- No extra `.env` keys are required for the speaker-ID fix.
- If you enrolled partner profiles before a speaker-ID backend change, reset the local browser data and record both partner samples again so the saved embeddings are regenerated together.
