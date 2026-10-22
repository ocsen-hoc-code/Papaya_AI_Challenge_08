"""Smoke test: run Qwen3-VL-8B-4bit (MLX) on one document and report speed and memory.

Usage: .venv/bin/python scripts/smoke_mlx.py [document] [--max-side 1600]
"""

from __future__ import annotations

import argparse
import math
import time

import fitz  # PyMuPDF
from PIL import Image

MODEL_ID = "mlx-community/Qwen3-VL-8B-Instruct-4bit"


def render(path: str, max_side: int) -> Image.Image:
    if path.lower().endswith(".pdf"):
        page = fitz.open(path)[0]
        pix = page.get_pixmap(dpi=200)
        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side), Image.LANCZOS)
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("document", nargs="?", default="document-tests/01_receipt_en.pdf")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_vlm import apply_chat_template, load, stream_generate

    t0 = time.perf_counter()
    model, processor = load(MODEL_ID)
    load_s = time.perf_counter() - t0

    image = render(args.document, args.max_side)
    messages = [
        {"role": "system", "content": "You read medical documents and answer with JSON only."},
        {
            "role": "user",
            "content": "Return JSON with keys document_type, hospital_or_lab_name, patient_name, date "
            "copied exactly as printed. Use null for anything not visible.",
        },
    ]
    prompt = apply_chat_template(processor, model.config, messages, num_images=1)

    text, logprobs, last = "", [], None
    t1 = time.perf_counter()
    for chunk in stream_generate(model, processor, prompt, image=[image], max_tokens=args.max_tokens, temperature=0.0):
        text += chunk.text
        last = chunk
        lp = chunk.logprobs
        if lp is not None and chunk.token is not None:
            # stream_generate yields the full log-softmax over the vocabulary per step.
            value = lp[chunk.token].item() if getattr(lp, "size", 1) > 1 else float(lp)
            logprobs.append(value)
    gen_s = time.perf_counter() - t1

    print("---- output ----")
    print(text)
    print("---- stats ----")
    print(f"image size        : {image.size}")
    print(f"load time         : {load_s:.1f}s")
    print(f"generation time   : {gen_s:.1f}s")
    print(f"prompt tokens     : {last.prompt_tokens}  ({last.prompt_tps:.1f} tok/s prefill)")
    print(f"generated tokens  : {last.generation_tokens}  ({last.generation_tps:.1f} tok/s)")
    print(f"peak memory       : {last.peak_memory:.2f} GB  (mx: {mx.get_peak_memory() / 1e9:.2f} GB)")
    if logprobs:
        mean_p = sum(math.exp(v) for v in logprobs) / len(logprobs)
        print(f"logprobs          : yes ({len(logprobs)} tokens, mean token prob {mean_p:.3f})")
    else:
        print("logprobs          : no")


if __name__ == "__main__":
    main()
