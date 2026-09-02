# Experiments

Nothing in this directory is part of the main application pipeline.

## Status

- `audio/tester.py`: standalone microphone/Faster Whisper prototype. The noise
  profile is never initialized during its normal execution, so denoising is
  effectively disabled.
- `audio/tester_debug.py`: unfinished audio-enhancement debugger. It records at
  16 kHz but resamples as though the source were 44.1 kHz, so its processing
  path needs correction before reuse.
- `ocr/screen_capture.py`: unfinished OCR/Donut experiment. Historical logs
  produced inconsistent extraction results; it is not connected to `main.py`.
- `ocr/samples/`: retained input image for reproducing the OCR experiment.
- `overlay/screen_overlay.py`: standalone Win32 overlay prototype, not connected
  to the assistant.
- `overlay/dll/dll.cpp`: source for a capture-exclusion DLL. The compiled build
  products were removed from the active branch and remain in the archive branch.
- `research/main.ipynb`: original research notebook and development history.
- `research/script.py`: an old notebook export with multiple incompatible
  prototypes and multiple entry points. Do not run it as the application.

Experimental dependencies are deliberately excluded from the root
`requirements.txt` so the main environment remains small and understandable.
