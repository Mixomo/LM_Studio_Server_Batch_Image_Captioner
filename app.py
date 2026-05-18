import base64
import inspect
import json
import mimetypes
import os
import queue
import re
import struct
import subprocess
import threading
import time
from importlib import metadata
from pathlib import Path
from tkinter import filedialog, messagebox, PanedWindow

import customtkinter as ctk


APP_TITLE = "llama.cpp Native Batch Image Captioner"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CONFIG_PATH = Path(__file__).with_name("user_config.json")
DEFAULT_CHAT_FORMAT = "llava-1-5"
DEFAULT_CONTEXT = "4096"
DEFAULT_GPU_LAYERS = "all"
DEFAULT_THREADS = ""
DEFAULT_MAX_TOKENS = "512"
DEFAULT_TEMPERATURE = "0.2"
DEFAULT_TOP_P = "0.9"
DEFAULT_PROMPT = """You are an expert image captioning AI specialized in generating highly detailed, accurate, and structured natural-language descriptions optimized for training and fine-tuning modern text-to-image models such as FLUX, Qwen Image, Z-Image, ERNIE Image, and similar architectures.

Your goal is to create captions that enable excellent prompt adherence, composition understanding, and strong concept, style, product, character, or identity learning during training. Always describe the image as if writing a precise prompt that another AI could use to faithfully reconstruct it.

Identity / subject trigger word:

(Here, enter the name or trigger word of the subject or concept you want to train)

Core Rules for Every Caption:

- Start with the main subject:
  Always begin by naming the primary subject using the provided trigger word / identity name first when one is provided.

- Trigger word integration:
  When a trigger word, identity name, character name, product name, or concept token is provided, integrate it naturally at the very beginning of the caption.
  Use this format:
  "{{TRIGGER_WORD}}, [description of visible appearance, action, pose, clothing, setting, lighting, and composition]"

- Do not repeat the subject type right after the trigger:
  The trigger word already represents the core identity, subject, character, product, or concept being trained.
  Do not immediately redefine it as "a young woman", "a man", "a person", "a character", "a dog", "a product", or another generic class unless the user explicitly requests class-word captions.

  Bad:
  "{{TRIGGER_WORD}}, a young woman with long dark hair..."
  "A medium shot of {{TRIGGER_WORD}}, a young woman with long, straight dark brown hair..."

  Good:
  "{{TRIGGER_WORD}}, with long straight dark brown hair, a radiant smile..."
  "A medium shot of {{TRIGGER_WORD}}, with long straight dark brown hair and a radiant smile..."

- Let the trigger own the identity:
  For identity, character, product, or concept training, let the trigger word carry the core identity, age, gender, base appearance, and recurring style. After the trigger, describe only visible details, variations, clothing, pose, expression, action, framing, lighting, environment, background, and other controllable image-specific elements.

- Use natural, fluent English:
  Write readable, human-like descriptive prose. Avoid pure tag lists and keyword spam, but use commas for clarity when listing attributes. The caption should feel like a dense visual prompt, not a story.

- Be comprehensive and dense:
  Describe visible content in high detail, including subjects, appearance, clothing and outfit, accessories, pose, action, facial expression, gaze direction, emotion if clearly visible, lighting, time of day if evident, environment, background, setting, atmosphere, colors, textures, materials, composition, perspective, camera angle, framing, depth of field, and any visible text, logos, or watermarks.

- Include technical and artistic aspects when evident:
  Mention photography style, art style, medium, rendering style, mood, lighting quality, and visual treatment when visible or strongly inferable from the image. Useful terms may include close-up portrait, medium shot, full-body shot, three-quarter view, profile view, candid photo, studio portrait, product photography, cinematic lighting, soft natural light, golden hour, volumetric light, dramatic shadows, shallow depth of field, bokeh background, sharp focus, photorealistic, anime style, 3D render, oil painting, watercolor, illustration, digital art, or realistic skin texture.

- Be precise and objective:
  Describe only what is clearly visible. Do not hallucinate unseen details, names, occupations, relationships, locations, camera models, brands, backstory, personality traits, or emotions that are not visually supported. If something is ambiguous, describe it conservatively.

- Handle multiple subjects clearly:
  If multiple subjects are visible, mention the trigger first when it is the main subject, then describe the other subjects and their spatial relationships, such as in the foreground, in the background, to the left, to the right, behind, beside, facing, holding, interacting with, or partially obscured by.

- Describe controllable variables:
  For LoRA, concept, character, product, or identity training, describe variables in detail so they remain controllable during generation. These include clothing, hairstyle changes if visible, accessories, facial expression, pose, action, camera framing, lighting, background, setting, props, color palette, and image style.

- Avoid over-describing fixed identity traits:
  Do not repeatedly overload every caption with the same complete identity description if the trigger word is meant to learn those traits. Include defining traits only when they are visible and useful, but keep the focus on the image-specific variation.

- Avoid subjective praise and filler:
  Do not use vague quality praise such as beautiful, gorgeous, perfect, amazing, masterpiece, best, ultra quality, or award-winning unless the word is literally visible as text in the image. Use visual evidence instead.

- Avoid contradictions and repetition:
  Do not contradict the image. Do not copy the exact same caption across images. Use consistent terminology for recurring features, but vary the image-specific details.

Caption length:
Write one rich, focused caption, typically 60 to 180 words depending on image complexity. Longer captions are acceptable for complex scenes and can help models like FLUX, Qwen, Z-Image, and ERNIE learn composition and prompt adherence, but keep the caption useful, non-repetitive, and grounded in visible content.

Output Format:
Provide ONLY the final caption text. Do not include introductions, explanations, labels such as "Caption:", markdown, bullet points, quotes, uncertainty notes, or meta commentary.

Now, analyze the provided image and the given trigger word, then generate the caption following all these guidelines."""
CHAT_FORMAT_HANDLERS = {
    "llava-1-5": "Llava15ChatHandler",
    "llava-1-6": "Llava16ChatHandler",
    "moondream2": "MoondreamChatHandler",
    "nanollava": "NanollavaChatHandler",
    "llama-3-vision-alpha": "Llama3VisionAlphaChatHandler",
    "minicpm-v-2.6": "MiniCPMv26ChatHandler",
    "qwen2.5-vl": "Qwen25VLChatHandler",
    "qwen3-vl": "Qwen3VLChatHandler",
    "gemma3": "Gemma3ChatHandler",
    "gemma4": "Gemma4ChatHandler",
}

GGUF_VALUE_TYPES = {
    0: "uint8",
    1: "int8",
    2: "uint16",
    3: "int16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "bool",
    8: "string",
    9: "array",
    10: "uint64",
    11: "int64",
    12: "float64",
}

GGUF_CONTEXT_KEYS = (
    "{arch}.context_length",
    "llama.context_length",
    "gemma.context_length",
    "gemma2.context_length",
    "gemma3.context_length",
    "gemma4.context_length",
    "qwen2vl.context_length",
    "qwen2_5_vl.context_length",
)


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
        self.geometry("1280x860")
        self.minsize(1040, 720)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.events = queue.Queue()
        self.image_paths = []
        self.worker = None
        self.llm = None
        self.loaded_config = None
        self.active_config = None
        self.active_prompt = DEFAULT_PROMPT

        self.model_path_var = ctk.StringVar(value="")
        self.mmproj_path_var = ctk.StringVar(value="")
        self.chat_format_var = ctk.StringVar(value=DEFAULT_CHAT_FORMAT)
        self.context_var = ctk.StringVar(value=DEFAULT_CONTEXT)
        self.gpu_layers_var = ctk.StringVar(value=DEFAULT_GPU_LAYERS)
        self.threads_var = ctk.StringVar(value=DEFAULT_THREADS)
        self.max_tokens_var = ctk.StringVar(value=DEFAULT_MAX_TOKENS)
        self.temperature_var = ctk.StringVar(value=DEFAULT_TEMPERATURE)
        self.top_p_var = ctk.StringVar(value=DEFAULT_TOP_P)
        self.overwrite_var = ctk.BooleanVar(value=False)

        self.build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
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
            text="Batch image captioning with llama-cpp-python directly in this app, without LM Studio as an intermediate server.",
            text_color=("gray35", "gray70"),
        )
        subtitle.grid(row=1, column=0, padx=18, pady=(0, 16), sticky="w")

        settings = ctk.CTkFrame(self)
        settings.grid(row=1, column=0, padx=18, pady=(18, 0), sticky="ew")
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(4, weight=1)

        self.add_label(settings, "Model GGUF", 0, 0)
        ctk.CTkEntry(settings, textvariable=self.model_path_var).grid(row=0, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(settings, text="Browse", width=90, command=self.browse_model).grid(row=0, column=2, padx=8, pady=8)

        self.add_label(settings, "mmproj", 1, 0)
        ctk.CTkEntry(settings, textvariable=self.mmproj_path_var).grid(row=1, column=1, padx=8, pady=8, sticky="ew")
        ctk.CTkButton(settings, text="Browse", width=90, command=self.browse_mmproj).grid(row=1, column=2, padx=8, pady=8)

        self.add_label(settings, "Chat format", 0, 3)
        ctk.CTkOptionMenu(settings, variable=self.chat_format_var, values=list(CHAT_FORMAT_HANDLERS)).grid(row=0, column=4, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "GPU layers", 1, 3)
        ctk.CTkEntry(settings, textvariable=self.gpu_layers_var, width=90).grid(row=1, column=4, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Context", 2, 0)
        ctk.CTkEntry(settings, textvariable=self.context_var).grid(row=2, column=1, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Threads", 2, 3)
        ctk.CTkEntry(settings, textvariable=self.threads_var).grid(row=2, column=4, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Max tokens", 3, 0)
        ctk.CTkEntry(settings, textvariable=self.max_tokens_var).grid(row=3, column=1, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Temperature", 3, 3)
        ctk.CTkEntry(settings, textvariable=self.temperature_var).grid(row=3, column=4, padx=8, pady=8, sticky="ew")

        self.add_label(settings, "Top P", 4, 0)
        ctk.CTkEntry(settings, textvariable=self.top_p_var).grid(row=4, column=1, padx=8, pady=8, sticky="ew")

        ctk.CTkButton(settings, text="Load Last Session", command=self.load_user_config).grid(row=5, column=0, padx=(12, 8), pady=(8, 12), sticky="ew")
        ctk.CTkButton(settings, text="Clear Prompt", command=self.clear_prompt).grid(row=5, column=1, padx=8, pady=(8, 12), sticky="ew")

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

        prompt_log_pane = PanedWindow(center, orient="vertical", sashwidth=8, showhandle=False, bd=0, bg="#1f1f1f")
        prompt_log_pane.grid(row=1, column=0, sticky="nsew")

        prompt_panel = ctk.CTkFrame(prompt_log_pane)
        prompt_panel.grid_columnconfigure(0, weight=1)
        prompt_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(prompt_panel, text="System Prompt", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=0, pady=(0, 6), sticky="w"
        )
        self.prompt_text = ctk.CTkTextbox(prompt_panel, height=180, wrap="word")
        self.prompt_text.grid(row=1, column=0, padx=0, pady=0, sticky="nsew")

        log_panel = ctk.CTkFrame(prompt_log_pane)
        log_panel.grid_columnconfigure(0, weight=1)
        log_panel.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(log_panel, text="Runtime Logs", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=0, padx=0, pady=(0, 6), sticky="w"
        )
        self.log_text = ctk.CTkTextbox(log_panel, wrap="word")
        self.log_text.grid(row=1, column=0, sticky="nsew")
        self.log_text.configure(state="disabled")

        prompt_log_pane.add(prompt_panel, minsize=120)
        prompt_log_pane.add(log_panel, minsize=220)
        self.after(250, lambda: prompt_log_pane.sash_place(0, 0, 220))

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

    def clear_prompt(self):
        self.prompt_text.delete("1.0", "end")
        self.log("Prompt cleared.")

    def load_user_config(self):
        if not CONFIG_PATH.exists():
            self.log("No previous session found.")
            return
        try:
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.log(f"Could not load user config: {exc}")
            return

        self.model_path_var.set(str(config.get("model_path", self.model_path_var.get())))
        self.mmproj_path_var.set(str(config.get("mmproj_path", self.mmproj_path_var.get())))
        self.chat_format_var.set(str(config.get("chat_format", self.chat_format_var.get())))
        self.context_var.set(str(config.get("context", self.context_var.get())))
        self.gpu_layers_var.set(str(config.get("gpu_layers", self.gpu_layers_var.get())))
        self.threads_var.set(str(config.get("threads", self.threads_var.get())))
        self.max_tokens_var.set(str(config.get("max_tokens", self.max_tokens_var.get())))
        self.temperature_var.set(str(config.get("temperature", self.temperature_var.get())))
        self.top_p_var.set(str(config.get("top_p", self.top_p_var.get())))
        self.overwrite_var.set(bool(config.get("overwrite", self.overwrite_var.get())))

        prompt = config.get("prompt")
        if isinstance(prompt, str) and prompt.strip():
            self.prompt_text.delete("1.0", "end")
            self.prompt_text.insert("1.0", prompt)

        geometry = config.get("geometry")
        if isinstance(geometry, str) and geometry:
            self.geometry(geometry)

        self.log("Loaded previous GUI configuration.")

    def save_user_config(self):
        config = {
            "model_path": self.model_path_var.get(),
            "mmproj_path": self.mmproj_path_var.get(),
            "chat_format": self.chat_format_var.get(),
            "context": self.context_var.get(),
            "gpu_layers": self.gpu_layers_var.get(),
            "threads": self.threads_var.get(),
            "max_tokens": self.max_tokens_var.get(),
            "temperature": self.temperature_var.get(),
            "top_p": self.top_p_var.get(),
            "overwrite": self.overwrite_var.get(),
            "prompt": self.prompt_text.get("1.0", "end").strip(),
            "geometry": self.geometry(),
        }
        CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")

    def on_close(self):
        try:
            self.save_user_config()
        except OSError as exc:
            self.log(f"Could not save user config: {exc}")
        self.destroy()

    def add_label(self, parent, text, row, column):
        ctk.CTkLabel(parent, text=text).grid(row=row, column=column, padx=(12, 4), pady=8, sticky="w")

    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select GGUF model",
            filetypes=[("GGUF model", "*.gguf"), ("All files", "*.*")],
        )
        if path:
            self.model_path_var.set(path)
            self.apply_model_metadata(Path(path))

    def browse_mmproj(self):
        path = filedialog.askopenfilename(
            title="Select multimodal projector",
            filetypes=[("GGUF or BIN projector", "*.gguf *.bin"), ("All files", "*.*")],
        )
        if path:
            self.mmproj_path_var.set(path)

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

        if not self.model_path_var.get().strip():
            messagebox.showwarning(APP_TITLE, "Select a vision-capable GGUF model first.")
            return

        model_path = self.resolve_model_path(self.model_path_var.get().strip())
        if model_path:
            self.model_path_var.set(str(model_path))
            self.apply_model_metadata(model_path)

        paths = self.filtered_paths()
        if not paths:
            messagebox.showinfo(APP_TITLE, "No images to caption. Enable overwrite to regenerate existing .txt files.")
            return

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.active_prompt = self.prompt_text.get("1.0", "end").strip()
        self.active_config = self.current_llm_config()
        try:
            self.save_user_config()
        except OSError as exc:
            self.log(f"Could not save user config: {exc}")
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
        config = self.active_config or self.current_llm_config()
        llm = self.get_llm()
        prompt = self.active_prompt

        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
        )
        caption = self.extract_caption(response)
        caption = self.clean_caption_output(caption)
        if not caption:
            raise RuntimeError(
                "llama.cpp returned an empty caption. "
                "Check that the GGUF model and mmproj match, and that the selected chat format supports your model."
            )
        return caption

    def get_llm(self):
        config = self.active_config or self.current_llm_config()
        model_config = self.model_config(config)
        if self.llm is not None and model_config == self.loaded_config:
            return self.llm

        model_path = Path(model_config["model_path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Model GGUF not found: {model_path}")

        mmproj_path = Path(model_config["mmproj_path"]) if model_config["mmproj_path"] else None
        if mmproj_path and not mmproj_path.exists():
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

        self.events.put(("log", "Loading llama.cpp model..."))
        self.configure_cuda_dll_paths()
        try:
            from llama_cpp import Llama
            from llama_cpp import llama_chat_format
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed in this environment. "
                "Run 1_install.bat, or install a matching CUDA wheel from the JamePeng release."
            ) from exc

        self.log_llamacpp_runtime()

        handler_name = CHAT_FORMAT_HANDLERS[model_config["chat_format"]]
        try:
            handler_class = getattr(llama_chat_format, handler_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Installed llama-cpp-python does not expose {handler_name}. "
                "Install a newer or model-specific wheel, such as the JamePeng CUDA release."
            ) from exc
        chat_handler = None
        if mmproj_path:
            handler_kwargs = self.chat_handler_kwargs(handler_class, str(mmproj_path))
            chat_handler = handler_class(**handler_kwargs)

        kwargs = {
            "model_path": str(model_path),
            "n_ctx": model_config["n_ctx"],
            "n_gpu_layers": model_config["n_gpu_layers"],
            "offload_kqv": True,
            "op_offload": True,
            "ctx_checkpoints": 0,
            "verbose": True,
            "verbosity": 3,
            "log_filters": [
                "llama_model_loader: - kv",
                "llama_model_loader: - type",
                "print_info:",
                "init_tokenizer:",
                "load: control",
                "load: special",
                "load: token",
                "load: printing",
            ],
        }
        if model_config["n_threads"] is not None:
            kwargs["n_threads"] = model_config["n_threads"]
        if chat_handler is not None:
            kwargs["chat_handler"] = chat_handler
        else:
            kwargs["chat_format"] = model_config["chat_format"]

        self.llm = Llama(**kwargs)
        self.loaded_config = model_config
        self.events.put(("log", f"Loaded {model_path.name} with n_gpu_layers={model_config['n_gpu_layers']}"))
        return self.llm

    def configure_cuda_dll_paths(self):
        paths = []
        try:
            dist = metadata.distribution("llama-cpp-python")
            package_lib = Path(dist.locate_file("")) / "llama_cpp" / "lib"
            paths.append(package_lib)
        except metadata.PackageNotFoundError:
            pass

        wheel_cuda_root = self.cuda_root_from_install_config()
        if wheel_cuda_root:
            paths.extend([wheel_cuda_root / "bin" / "x64", wheel_cuda_root / "bin"])

        for env_name in ("CUDA_PATH", "CUDA_HOME"):
            cuda_root = os.environ.get(env_name)
            if cuda_root:
                paths.extend([
                    Path(cuda_root) / "bin" / "x64",
                    Path(cuda_root) / "bin",
                ])

        cuda_base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if cuda_base.exists():
            for cuda_root in sorted(cuda_base.glob("v*"), reverse=True):
                paths.extend([cuda_root / "bin" / "x64", cuda_root / "bin"])

        added = []
        for path in paths:
            if not path.exists():
                continue
            path_text = str(path)
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(path_text)
                except OSError:
                    pass
            if path_text not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path_text + os.pathsep + os.environ.get("PATH", "")
            added.append(path_text)

        cublas_name = self.cublas_dll_name_from_install_config()
        cublas = self.find_on_path(cublas_name)
        self.events.put(("log", "DLL search paths added: " + "; ".join(added[:6])))
        self.events.put(("log", f"{cublas_name} visible: {cublas or 'no'}"))

    def cublas_dll_name_from_install_config(self):
        wheel_cuda = self.wheel_cuda_from_install_config()
        match = re.fullmatch(r"cu(\d+)", wheel_cuda or "")
        if not match:
            return "cublas64_13.dll"
        digits = match.group(1)
        major = digits[:2] if len(digits) >= 2 else digits
        return f"cublas64_{major}.dll"

    def cuda_root_from_install_config(self):
        wheel_cuda = self.wheel_cuda_from_install_config()
        match = re.fullmatch(r"cu(\d+)", wheel_cuda or "")
        if not match:
            return None

        digits = match.group(1)
        candidates = []
        if len(digits) >= 3:
            candidates.append(f"v{digits[:-1]}.{digits[-1]}")
        if len(digits) >= 2:
            candidates.append(f"v{digits[:2]}.{digits[2:] or '0'}")
            candidates.append(f"v{digits[:2]}.0")

        cuda_base = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        for name in candidates:
            root = cuda_base / name
            if root.exists():
                return root
        return None

    def wheel_cuda_from_install_config(self):
        config_path = Path(__file__).with_name(".llama_cpp_cuda.json")
        if not config_path.exists():
            return ""
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return ""
        return str(config.get("wheel_cuda") or "")

    def find_on_path(self, filename):
        for item in os.environ.get("PATH", "").split(os.pathsep):
            if not item:
                continue
            candidate = Path(item) / filename
            if candidate.exists():
                return str(candidate)
        return ""

    def log_llamacpp_runtime(self):
        try:
            dist = metadata.distribution("llama-cpp-python")
        except metadata.PackageNotFoundError:
            self.events.put(("log", "llama-cpp-python distribution metadata not found."))
            return

        location = Path(dist.locate_file(""))
        direct_url_path = Path(dist._path) / "direct_url.json"
        source = "unknown"
        if direct_url_path.exists():
            try:
                source = json.loads(direct_url_path.read_text(encoding="utf-8")).get("url", source)
            except (OSError, json.JSONDecodeError):
                pass

        cuda_dll = location / "llama_cpp" / "lib" / "ggml-cuda.dll"
        self.events.put(("log", f"llama-cpp-python {dist.version}"))
        self.events.put(("log", f"llama-cpp-python source: {source}"))
        self.events.put(("log", f"ggml-cuda.dll present: {cuda_dll.exists()}"))

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            gpu_info = (result.stdout or result.stderr).strip()
            if gpu_info:
                self.events.put(("log", f"nvidia-smi: {gpu_info}"))
        except (OSError, subprocess.SubprocessError):
            self.events.put(("log", "nvidia-smi not available from this process."))

    def model_config(self, config):
        return {
            "model_path": config["model_path"],
            "mmproj_path": config["mmproj_path"],
            "chat_format": config["chat_format"],
            "n_ctx": config["n_ctx"],
            "n_gpu_layers": config["n_gpu_layers"],
            "n_threads": config["n_threads"],
        }

    def current_llm_config(self):
        model_path = self.resolve_model_path(self.model_path_var.get().strip())
        return {
            "model_path": str(model_path) if model_path else self.model_path_var.get().strip(),
            "mmproj_path": self.mmproj_path_var.get().strip(),
            "chat_format": self.chat_format_var.get().strip() or DEFAULT_CHAT_FORMAT,
            "n_ctx": self.parse_int(self.context_var.get(), int(DEFAULT_CONTEXT)),
            "n_gpu_layers": self.parse_gpu_layers(self.gpu_layers_var.get()),
            "n_threads": self.parse_optional_int(self.threads_var.get()),
            "max_tokens": self.parse_int(self.max_tokens_var.get(), int(DEFAULT_MAX_TOKENS)),
            "temperature": self.parse_float(self.temperature_var.get(), float(DEFAULT_TEMPERATURE)),
            "top_p": self.parse_float(self.top_p_var.get(), float(DEFAULT_TOP_P)),
        }

    def chat_handler_kwargs(self, handler_class, clip_model_path):
        kwargs = {"clip_model_path": clip_model_path}
        try:
            signature = inspect.signature(handler_class.__init__)
        except (TypeError, ValueError):
            return kwargs

        parameters = signature.parameters
        if "enable_thinking" in parameters:
            kwargs["enable_thinking"] = False
        if "force_reasoning" in parameters:
            kwargs["force_reasoning"] = False
        if "preserve_thinking" in parameters:
            kwargs["preserve_thinking"] = False
        if "keep_past_thinking" in parameters:
            kwargs["keep_past_thinking"] = False
        return kwargs

    def clean_caption_output(self, text):
        text = self.strip_channel_thought_blocks(text)
        text = re.sub(r"(?is)<think>.*?</think>", "", text)
        text = re.sub(
            r"(?i)<\|?/?(?:channel|thought|analysis|final|assistant|system|user|turn|start|end|eos|bos|pad|image|audio|video)[^>]*\|?>",
            "",
            text,
        )
        text = re.sub(r"(?m)^\s*[`*_#>\-]+(?:\s|$)", "", text)
        text = text.replace("`", "")
        text = re.sub(r"(?im)^\s*(thought|analysis|final)\s*$", "", text)
        text = re.sub(r"(?m)^[ \t]*<[|/]?[^>\n]{1,64}[|]?>[ \t]*", "", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def strip_channel_thought_blocks(self, text):
        text = re.sub(r"(?is)<\|channel\>\s*thought\s*\n?.*?<channel\|>", "", text)
        text = re.sub(r"(?is)<\|channel\>\s*analysis\s*\n?.*?<channel\|>", "", text)
        text = re.sub(r"(?is)<\|channel\>\s*commentary\s*\n?.*?<channel\|>", "", text)
        text = re.sub(r"(?i)<\|channel\>\s*final\s*\n?", "", text)
        if "<channel|>" in text:
            text = text.split("<channel|>")[-1]
        return text

    def resolve_model_path(self, value):
        if not value:
            return None
        path = Path(value)
        if path.is_file():
            return path
        if not path.is_dir():
            return None
        candidates = [
            item for item in path.glob("*.gguf")
            if not item.name.lower().startswith("mmproj")
        ]
        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item.stat().st_size, reverse=True)[0]

    def apply_model_metadata(self, model_path):
        try:
            metadata = self.read_gguf_metadata(model_path)
        except Exception as exc:
            self.events.put(("log", f"Could not read GGUF metadata: {exc}"))
            return

        architecture = str(metadata.get("general.architecture", "")).lower()
        context_length = self.infer_context_length(metadata, architecture)

        chat_format = self.infer_chat_format(metadata, architecture, model_path)
        if chat_format in CHAT_FORMAT_HANDLERS:
            self.chat_format_var.set(chat_format)

        if not self.mmproj_path_var.get().strip():
            mmproj_path = self.find_mmproj(model_path)
            if mmproj_path:
                self.mmproj_path_var.set(str(mmproj_path))

        details = []
        if architecture:
            details.append(f"architecture={architecture}")
        if context_length:
            details.append(f"trained_context={context_length}")
        if chat_format:
            details.append(f"chat_format={chat_format}")
        if details:
            self.events.put(("log", "Read GGUF metadata: " + ", ".join(details)))

    def read_gguf_metadata(self, model_path):
        with model_path.open("rb") as file:
            if file.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            version = self.read_struct(file, "<I")
            if version not in {2, 3}:
                raise ValueError(f"unsupported GGUF version: {version}")
            tensor_count = self.read_struct(file, "<Q")
            metadata_count = self.read_struct(file, "<Q")

            metadata = {}
            for _ in range(metadata_count):
                key = self.read_gguf_string(file)
                value_type = self.read_struct(file, "<I")
                metadata[key] = self.read_gguf_value(file, value_type)

            metadata["_tensor_count"] = tensor_count
            metadata["_version"] = version
            return metadata

    def read_gguf_value(self, file, value_type):
        if value_type == 0:
            return self.read_struct(file, "<B")
        if value_type == 1:
            return self.read_struct(file, "<b")
        if value_type == 2:
            return self.read_struct(file, "<H")
        if value_type == 3:
            return self.read_struct(file, "<h")
        if value_type == 4:
            return self.read_struct(file, "<I")
        if value_type == 5:
            return self.read_struct(file, "<i")
        if value_type == 6:
            return self.read_struct(file, "<f")
        if value_type == 7:
            return bool(self.read_struct(file, "<?"))
        if value_type == 8:
            return self.read_gguf_string(file)
        if value_type == 9:
            item_type = self.read_struct(file, "<I")
            item_count = self.read_struct(file, "<Q")
            item_name = GGUF_VALUE_TYPES.get(item_type)
            if item_name is None:
                raise ValueError(f"unsupported GGUF array type: {item_type}")
            if item_type == 8:
                preview = []
                for index in range(item_count):
                    value = self.read_gguf_string(file)
                    if index < 16:
                        preview.append(value)
                return preview if item_count <= 16 else f"<string[{item_count}]>"
            item_size = self.gguf_scalar_size(item_type)
            if item_size is None:
                raise ValueError(f"unsupported GGUF array item type: {item_type}")
            file.seek(item_size * item_count, 1)
            return f"<{item_name}[{item_count}]>"
        if value_type == 10:
            return self.read_struct(file, "<Q")
        if value_type == 11:
            return self.read_struct(file, "<q")
        if value_type == 12:
            return self.read_struct(file, "<d")
        raise ValueError(f"unsupported GGUF value type: {value_type}")

    def read_gguf_string(self, file):
        length = self.read_struct(file, "<Q")
        return file.read(length).decode("utf-8", errors="replace")

    def read_struct(self, file, fmt):
        size = struct.calcsize(fmt)
        data = file.read(size)
        if len(data) != size:
            raise EOFError("unexpected end of GGUF metadata")
        return struct.unpack(fmt, data)[0]

    def gguf_scalar_size(self, value_type):
        return {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 4,
            5: 4,
            6: 4,
            7: 1,
            10: 8,
            11: 8,
            12: 8,
        }.get(value_type)

    def infer_context_length(self, metadata, architecture):
        for key_template in GGUF_CONTEXT_KEYS:
            key = key_template.format(arch=architecture)
            value = metadata.get(key)
            if isinstance(value, int) and value > 0:
                return value
        for key, value in metadata.items():
            if key.endswith(".context_length") and isinstance(value, int) and value > 0:
                return value
        return None

    def infer_chat_format(self, metadata, architecture, model_path):
        name = model_path.name.lower()
        chat_template = str(metadata.get("tokenizer.chat_template", "")).lower()
        haystack = f"{architecture} {name} {chat_template}"
        if "gemma4" in haystack or "gemma-4" in haystack or "gemma_4" in haystack:
            return "gemma4"
        if "gemma3" in haystack or "gemma-3" in haystack or "gemma_3" in haystack:
            return "gemma3"
        if "qwen3" in haystack and "vl" in haystack:
            return "qwen3-vl"
        if "qwen2.5" in haystack and "vl" in haystack:
            return "qwen2.5-vl"
        if "minicpm" in haystack:
            return "minicpm-v-2.6"
        if "llava" in haystack and ("1.6" in haystack or "v1.6" in haystack):
            return "llava-1-6"
        if "llava" in haystack:
            return "llava-1-5"
        return self.chat_format_var.get().strip() or DEFAULT_CHAT_FORMAT

    def find_mmproj(self, model_path):
        candidates = []
        for pattern in ("mmproj*.gguf", "*mmproj*.gguf", "mmproj*.bin", "*mmproj*.bin"):
            candidates.extend(model_path.parent.glob(pattern))
        candidates = [item for item in candidates if item.is_file()]
        if not candidates:
            return None
        model_tokens = set(model_path.stem.lower().replace("_", "-").split("-"))

        def score(path):
            path_tokens = set(path.stem.lower().replace("_", "-").split("-"))
            return (len(model_tokens & path_tokens), path.stat().st_size)

        return sorted(candidates, key=score, reverse=True)[0]

    def parse_int(self, value, default):
        try:
            return int(value.strip())
        except (AttributeError, TypeError, ValueError):
            return default

    def parse_gpu_layers(self, value):
        value = (value or "").strip().lower()
        if value in {"", "all", "auto"}:
            return value or DEFAULT_GPU_LAYERS
        return self.parse_int(value, -1)

    def parse_optional_int(self, value):
        value = value.strip()
        if not value:
            return None
        return self.parse_int(value, 0)

    def parse_float(self, value, default):
        try:
            return float(value.strip())
        except (AttributeError, TypeError, ValueError):
            return default

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
        self.log_text.configure(state="normal")
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

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
