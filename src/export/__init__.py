"""Export GGUF + Ollama theo SOLID: mỗi module một trách nhiệm.

- proc: chạy tiến trình con, stream log realtime qua callback.
- ollama_env: đảm bảo Ollama CLI + server (tự cài trên Colab/Linux).
- bundle: merge adapter + convert GGUF, verify đủ mmproj/Modelfile.
- ollama_import: nạp bundle vào Ollama (``ollama create``).

UI (scripts/ui.py) chỉ wiring Gradio, không chứa logic export.
"""
