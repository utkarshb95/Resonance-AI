# Resonance AI

A Windows meeting-assistant prototype that captures microphone and system
audio, transcribes speech with Faster Whisper, detects questions, and generates
short answers through Groq.

## Current baseline

The runnable application is the five-module pipeline in the repository root:

1. `main.py` starts and stops the application.
2. `audio_stream.py` captures microphone and system audio.
3. `audio_processing.py` filters, transcribes, and deduplicates speech.
4. `response.py` detects questions and generates responses.
5. `config.py` contains models, queues, audio devices, and tuning values.

These files were restored from Git commit `d61c463`, the last revision recorded
as working. The configured Windows audio-device indexes and Groq model IDs were
then refreshed against this machine and account on September 2, 2026.

## Before running

- Use Python 3.12 in a dedicated virtual or Conda environment.
- Install `requirements.txt`.
- Set the `GROQ_API_KEY` environment variable.
- Confirm the input/output indexes in `config.py`; they are machine-specific.
- The current Whisper configuration requires a CUDA-capable NVIDIA GPU.

Run the application from the repository root:

```powershell
python main.py
```

## Repository layout

```text
.
|-- main.py                     # application entry point
|-- config.py                   # application configuration
|-- audio_stream.py             # live audio capture
|-- audio_processing.py         # transcription pipeline
|-- response.py                 # question/answer pipeline
|-- requirements.txt            # core application dependencies
|-- tests/
|   `-- response_integration_check.py
`-- experiments/
    |-- audio/                   # standalone capture/denoising prototypes
    |-- ocr/                     # unfinished screen-reading prototype
    |-- overlay/                 # unfinished Windows overlay prototype
    `-- research/                # notebook and exported exploratory script
```

The integration check is intentionally manual: it loads ML models and may call
the Groq API. It is not a fast or offline unit test.

## Recovery

The complete pre-cleanup working tree—including unfinished rewrites, duplicate
backup files, generated OCR output, and DLL build products—is preserved in the
local branch `archive/pre-cleanup-2026-09-02` at commit `cd44bee`. Nothing was
pushed to a remote.
