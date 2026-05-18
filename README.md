# llama.cpp Native Batch Image Captioner

Minimal CustomTkinter app for captioning many images directly through `llama-cpp-python`, without running LM Studio as a local server. Each image is sent as a separate request, so context does not accumulate across the batch.

![GUI](assets/GUI.png)

## Install

Double-click:

```text
1_install.bat
```

The installer uses `uv`. If `uv` is missing, it tries to install it automatically with `winget`, then runs `uv sync`.

After that it installs `llama-cpp-python` as automatically as possible:

1. Detects the Python wheel tag used by `uv`, such as `cp310`, `cp311`, or `cp312`.
2. Detects the CUDA version from `nvidia-smi`, `nvcc`, `CUDA_PATH`, or `CUDA_HOME`.
3. Queries the latest JamePeng releases from:

```text
https://github.com/JamePeng/llama-cpp-python/releases/
```

4. Installs the best matching Windows `win_amd64` wheel directly with `uv pip install <wheel-url>`.

Manual/offline installs are still supported. Put a wheel in:

```text
wheels\llama_cpp_python*.whl
```

Or set:

```text
LLAMA_CPP_PYTHON_WHEEL=C:\path\to\llama_cpp_python.whl
```

If CUDA or a matching release cannot be detected, the installer falls back to the standard PyPI `llama-cpp-python` package. On Windows this may compile from source and can be slower or fail if build tools/CUDA are missing.

## Run

Double-click:

```text
2_run.bat
```

The runner uses `uv run --no-sync` so the CUDA wheel installed by `1_install.bat` is not replaced or removed by a later project sync.

## Model Setup

Use a vision-capable GGUF model and, when the model requires it, the matching multimodal projector file (`mmproj`).

Common fields:

```text
Model GGUF:  path to the main .gguf model, selected with Browse
mmproj:      path to the matching mmproj .gguf or .bin file, selected with Browse
Chat format: model family used by llama-cpp-python
GPU layers:  all to offload all possible layers to GPU
Context:     prompt context size
Threads:     empty lets llama.cpp choose
Max tokens:  maximum caption length
```

When a model loads, the log prints the installed `llama-cpp-python` version, the source wheel URL, whether `ggml-cuda.dll` is present, and basic `nvidia-smi` GPU information. llama.cpp verbose loading logs are enabled so you can see whether layers are actually offloaded to CUDA.

The app reads GGUF metadata to infer architecture, chat format, and matching `mmproj`, but it does not overwrite `Context`. The default runtime context stays at `4096` even when the GGUF reports a much larger trained context.

Supported chat format presets in the app:

```text
llava-1-5
llava-1-6
moondream2
nanollava
llama-3-vision-alpha
minicpm-v-2.6
qwen2.5-vl
qwen3-vl
```

The model and `mmproj` must belong together. If captions are empty or the image is ignored, the most common causes are a mismatched projector, wrong chat format, or a model build without the needed multimodal support.

## Output

The app writes one `.txt` caption next to each image:

```text
photo_001.jpg
photo_001.txt
```

Use `Overwrite existing .txt files to start fresh captioning` if you want to regenerate captions. Leave it unchecked to skip images that already have matching `.txt` files.

## System Prompt

The text box in the app is the caption prompt that used to live in the LM Studio preset. Edit it before starting a batch if you want a different captioning style, trigger-word behavior, or output length.

The prompt and log panes are vertically resizable. Drag the divider between them to give more space to either the prompt or runtime logs.

Saved captions are cleaned of common model control tokens such as `<|channel>`, `<channel|>`, `<think>...</think>`, and leading `thought` / `analysis` markers.

The app saves the last GUI configuration to `user_config.json` when you start a batch or close the window. New launches start empty; use `Load Last Session` to restore the previous session. Use `Clear Prompt` to empty the prompt box.

## Example System Prompt:

You can ask a high-capacity LLM (Gemini, Grok, ChatGPT, Claude, etc.) to generate a custom system prompt for you

Here is an example system prompt (tested with google/gemma-4-26b-a4b model -  Other models may vary in their output) : 

```text
You are an expert image captioning AI specialized in generating highly detailed, accurate, and structured natural-language descriptions optimized for training and fine-tuning modern text-to-image models such as FLUX, Qwen Image, Z-Image, ERNIE Image, and similar architectures.

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

Now, analyze the provided image and the given trigger word, then generate the caption following all these guidelines.
```
