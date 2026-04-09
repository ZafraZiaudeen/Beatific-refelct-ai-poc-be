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

5. Warm the SpeechBrain speaker model into the local cache before serving requests:

```powershell
cd python-service
.\venv\Scripts\python.exe prepare_speaker_model.py
```

Git Bash equivalent:

```bash
cd python-service
./venv/Scripts/python.exe prepare_speaker_model.py
```

6. Start the Python speaker ID service with the same project virtualenv:

```powershell
cd python-service
.\venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8200
```

Git Bash equivalent:

```bash
cd python-service
./venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8200
```

7. Start the Node backend:

```powershell
npm run dev
```

The frontend in `../reflect-ai-poc` should point to `http://localhost:3100`.
