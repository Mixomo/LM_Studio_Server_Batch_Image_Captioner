import base64
import mimetypes
import queue
import threading
import time
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import requests


APP_TITLE = "LM Studio Server Batch Image Captioner"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_API_BASE = "http://127.0.0.1:1234"
DEFAULT_CHAT_URL = f"{DEFAULT_API_BASE}/v1/chat/completions"
DEFAULT_MODELS_URL = f"{DEFAULT_API_BASE}/v1/models"
DEFAULT_MODEL = "google/gemma-4-26b-a4b"


class CaptionWorker(threading.Thread):
    def __init__(self, app, image_paths):
        super().__init__(daemon=True)
        self.app = app
        self.image_paths = image_paths
        self.stop_requested = threading.Event()

    def run(self):
        total = len(self.image_paths)
        start_time = time.monotonic()
        for index, image_path in enumerate(self.image_paths, start=1):
            if self.stop_requested.is_set():
                self.app.events.put(("log", "Stopped by user."))
                break

            completed = index - 1
            eta_seconds = None
            if completed:
                elapsed = time.monotonic() - start_time
                eta_seconds = (elapsed / completed) * (total - completed)
            self.app.events.put(("progress", completed, total, eta_seconds))
            self.app.events.put(("status", f"Captioning {index}/{total}: {image_path.name}"))

            try:
                caption = self.app.caption_image(image_path)
                output_path = image_path.with_suffix(".txt")
                output_path.write_text(caption.strip() + "\n", encoding="utf-8")
                self.app.events.put(("log", f"OK  {image_path.name} -> {output_path.name}"))
            except Exception as exc:
                self.app.events.put(("log", f"ERR {image_path.name}: {exc}"))

            elapsed = time.monotonic() - start_time
            average = elapsed / index
            remaining = max(total - index, 0)
            self.app.events.put(("progress", index, total, average * remaining))

        self.app.events.put(("progress", total, total, 0))
        self.app.events.put(("done",))


class BatchCaptionerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x620")
        self.minsize(980, 500)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.events = queue.Queue()
        self.image_paths = []
        self.worker = None

        self.api_base_var = ctk.StringVar(value=DEFAULT_API_BASE)
        self.chat_url_var = ctk.StringVar(value=DEFAULT_CHAT_URL)
        self.models_url_var = ctk.StringVar(value=DEFAULT_MODELS_URL)
        self.model_var = ctk.StringVar(value=DEFAULT_MODEL)
        self.overwrite_var = ctk.BooleanVar(value=False)

        self.build_ui()
        self.after(100, self.process_events)

    def build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title = ctk.CTkLabel(header, text=APP_TITLE, font=ctk.CTkFont(size=22, weight="bold"))
        title.grid(row=0, column=0, padx=18, pady=(16, 4), sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Batch image captioning to create datasets for training text-to-image models using vision LLMs via the LM Studio server. Fill in or replace the fields as shown in LM Studio.",
            text_color=("gray35", "gray70"),
        )
        subtitle.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")

        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, padx=18, pady=(18, 0), sticky="ew")
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)

        self.add_label(settings, "API base", 0, 0)
        ctk.CTkEntry(settings, textvariable=self.api_base_var).grid(row=0, column=1, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Model", 0, 2)
        ctk.CTkEntry(settings, textvariable=self.model_var).grid(row=0, column=3, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Chat URL", 1, 0)
        ctk.CTkEntry(settings, textvariable=self.chat_url_var).grid(row=1, column=1, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Models URL", 1, 2)
        ctk.CTkEntry(settings, textvariable=self.models_url_var).grid(row=1, column=3, padx=8, pady=8, sticky="ew")

        center = ctk.CTkFrame(self)
        center.grid(row=2, column=0, padx=18, pady=18, sticky="nsew")
        center.grid_columnconfigure(0, weight=1)
        center.grid_rowconfigure(1, weight=1)

        controls = ctk.CTkFrame(center, fg_color="transparent")
        controls.grid(row=0, column=0, sticky="ew")
        controls.grid_columnconfigure(3, weight=1)

        ctk.CTkButton(controls, text="Add Images", command=self.add_images).grid(row=0, column=0, padx=(0, 8), pady=8)
        ctk.CTkButton(controls, text="Add Folder", command=self.add_folder).grid(row=0, column=1, padx=8, pady=8)
        ctk.CTkButton(controls, text="Clear", command=self.clear_images).grid(row=0, column=2, padx=8, pady=8)
        ctk.CTkCheckBox(controls, text="Overwrite existing .txt files to start fresh captioning", variable=self.overwrite_var).grid(row=0, column=3, padx=8, pady=8)
        self.start_button = ctk.CTkButton(controls, text="Start", command=self.start_captioning)
        self.start_button.grid(row=0, column=4, padx=8, pady=8)
        self.stop_button = ctk.CTkButton(controls, text="Stop", command=self.stop_captioning, state="disabled")
        self.stop_button.grid(row=0, column=5, padx=(8, 0), pady=8)

        self.log_text = ctk.CTkTextbox(center, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew")

        footer = ctk.CTkFrame(self, corner_radius=0)
        footer.grid(row=4, column=0, sticky="ew")
        footer.grid_columnconfigure(0, weight=1)

        self.progress = ctk.CTkProgressBar(footer)
        self.progress.grid(row=0, column=0, padx=18, pady=(12, 4), sticky="ew")
        self.progress.set(0)

        self.progress_label = ctk.CTkLabel(footer, text="Progress: 0/0 | ETA: --:--", text_color=("gray35", "gray70"))
        self.progress_label.grid(row=1, column=0, padx=18, pady=(0, 2), sticky="w")

        self.status_label = ctk.CTkLabel(footer, text="Ready", text_color=("gray35", "gray70"))
        self.status_label.grid(row=2, column=0, padx=18, pady=(0, 12), sticky="w")

    def add_label(self, parent, text, row, column):
        ctk.CTkLabel(parent, text=text).grid(row=row, column=column, padx=(12, 4), pady=8, sticky="w")

    def add_images(self):
        files = filedialog.askopenfilenames(
            title="Select images",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"), ("All files", "*.*")],
        )
        self.add_paths([Path(file) for file in files])

    def add_folder(self):
        folder = filedialog.askdirectory(title="Select image folder")
        if not folder:
            return
        paths = [path for path in Path(folder).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        self.add_paths(paths)

    def add_paths(self, paths):
        existing = set(self.image_paths)
        added = []
        for path in paths:
            if path.suffix.lower() in IMAGE_EXTENSIONS and path not in existing:
                self.image_paths.append(path)
                existing.add(path)
                added.append(path)
        self.log(f"Added {len(added)} image(s). Total: {len(self.image_paths)}")

    def clear_images(self):
        self.image_paths.clear()
        self.log_text.delete("1.0", "end")
        self.status_label.configure(text="Ready")
        self.progress_label.configure(text="Progress: 0/0 | ETA: --:--")
        self.progress.set(0)

    def start_captioning(self):
        if not self.image_paths:
            messagebox.showwarning(APP_TITLE, "Add images or a folder first.")
            return

        paths = self.filtered_paths()
        if not paths:
            messagebox.showinfo(APP_TITLE, "No images to caption. Enable overwrite to regenerate existing .txt files.")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker = CaptionWorker(self, paths)
        self.worker.start()

    def filtered_paths(self):
        if self.overwrite_var.get():
            return list(self.image_paths)
        return [path for path in self.image_paths if not path.with_suffix(".txt").exists()]

    def stop_captioning(self):
        if self.worker:
            self.worker.stop_requested.set()
            self.status_label.configure(text="Stopping after current image...")

    def caption_image(self, image_path):
        image_data_url = self.image_to_data_url(image_path)

        payload = {
            "model": self.get_model_id(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        }

        response = requests.post(self.chat_url_var.get().strip() or DEFAULT_CHAT_URL, json=payload, timeout=180)
        response.raise_for_status()
        data = response.json()
        caption = self.extract_caption(data)
        if not caption:
            raise RuntimeError(
                "LM Studio returned an empty caption. "
                "Check that the selected model is vision-capable and that the LM Studio preset is active."
            )
        return caption

    def get_model_id(self):
        model_id = self.model_var.get().strip()
        if model_id:
            return model_id

        try:
            response = requests.get(self.models_url_var.get().strip() or DEFAULT_MODELS_URL, timeout=10)
            response.raise_for_status()
            models = response.json().get("data") or []
            if models and models[0].get("id"):
                self.events.put(("log", f"Using model: {models[0]['id']}"))
                return models[0]["id"]
        except Exception as exc:
            self.events.put(("log", f"Could not auto-detect model, using default: {exc}"))

        return DEFAULT_MODEL

    def extract_caption(self, data):
        if isinstance(data.get("output_text"), str):
            return data["output_text"].strip()

        choices = data.get("choices") or []
        if not choices:
            return ""

        choice = choices[0]
        if isinstance(choice.get("text"), str):
            return choice["text"].strip()

        message = choice.get("message") or {}
        content = message.get("content")

        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    if isinstance(text, str):
                        parts.append(text)
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts).strip()

        return ""

    def image_to_data_url(self, image_path):
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")

    def process_events(self):
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "log":
                    self.log(event[1])
                elif kind == "status":
                    self.status_label.configure(text=event[1])
                elif kind == "progress":
                    current, total, eta_seconds = event[1], event[2], event[3]
                    self.progress.set(0 if total == 0 else current / total)
                    self.progress_label.configure(text=self.format_progress(current, total, eta_seconds))
                elif kind == "done":
                    self.start_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    self.status_label.configure(text="Done")
        except queue.Empty:
            pass
        self.after(100, self.process_events)

    def format_progress(self, current, total, eta_seconds):
        if total == 0:
            return "Progress: 0/0 | ETA: --:--"
        if eta_seconds is None:
            eta_text = "calculating..."
        else:
            eta_text = self.format_duration(eta_seconds)
        return f"Progress: {current}/{total} | ETA: {eta_text}"

    def format_duration(self, seconds):
        seconds = max(int(round(seconds)), 0)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    app = BatchCaptionerApp()
    app.mainloop()
