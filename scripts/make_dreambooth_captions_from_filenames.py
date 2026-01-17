#!/usr/bin/env python3
"""
Create DreamBooth-style caption .txt files next to images, using Florence-2 via Replicate.

Example:
  dataset/ava/ava_front_view.png
  -> dataset/ava/ava_front_view.txt containing:
     girl, ava1, <florence_caption>
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
FLORENCE2_MODEL = (
    "lucataco/florence-2-large:"
    "da53547e17d45b9cfb48174b2f18af8b83ca020fa76db62136bf9c6616762595"
)


def _load_dotenv_if_present(dotenv_path: Path) -> None:
    """
    Minimal .env loader (so we don't need an extra dependency).
    Loads KEY=VALUE pairs into os.environ if the key is not already set.
    """

    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _iter_images(dataset_dir: Path, exts: Iterable[str] = IMAGE_EXTS) -> list[Path]:
    exts_lc = {e.lower() for e in exts}
    images: list[Path] = []
    for p in sorted(dataset_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts_lc:
            images.append(p)
    return images


def _maybe_parse_structured_string(s: str) -> Any:
    """
    Replicate sometimes returns structured data; if we end up with it stringified
    (e.g., \"{'<CAPTION>': '...'}\"), try to parse it back into a Python object.
    """

    raw = s.strip()
    if not raw:
        return s

    # Try JSON first (double-quoted keys/strings)
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return json.loads(raw)
        except Exception:
            pass

    # Try Python literal (single quotes, dict repr)
    if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass

    return s


def _extract_caption_text(output: object) -> str:
    """
    Replicate model outputs vary. Try common patterns robustly.
    """
    if output is None:
        return ""
    if isinstance(output, bytes):
        try:
            return output.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""
    if isinstance(output, str):
        parsed = _maybe_parse_structured_string(output)
        if parsed is not output:
            return _extract_caption_text(parsed)
        return output.strip()

    # Dict-shaped outputs
    if isinstance(output, dict):
        d: dict[Any, Any] = output

        # Florence-2 schema: output object has "text" (string) and "img" (uri).
        # In practice, "text" sometimes contains structured task-keyed results
        # like {"<CAPTION>": "..."} or even a stringified dict.
        # See: https://replicate.com/lucataco/florence-2-large/api/schema#output-schema
        if "text" in d:
            nested = _extract_caption_text(d.get("text"))
            if nested:
                return nested

        # Task-keyed outputs (commonly looks like {"<CAPTION>": "..."}).
        # Prefer CAPTION, then DETAILED variants if present.
        preferred_task_keys = (
            "<CAPTION>",
            "<DETAILED_CAPTION>",
            "<MORE_DETAILED_CAPTION>",
            "<MORE_DETAILED_DESCRIPTION>",
            "<OD>",
            "<OCR>",
        )
        for k in preferred_task_keys:
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (dict, list, tuple)):
                nested = _extract_caption_text(v)
                if nested:
                    return nested

        # If any "<...>" key exists with a string value, pick the first (stable order by key name)
        angle_keys = sorted(
            (kk for kk in d.keys() if isinstance(kk, str) and kk.startswith("<") and kk.endswith(">")),
            key=str,
        )
        for kk in angle_keys:
            vv = d.get(kk)
            if isinstance(vv, str) and vv.strip():
                return vv.strip()
            if isinstance(vv, (dict, list, tuple)):
                nested = _extract_caption_text(vv)
                if nested:
                    return nested

        # Other common keys
        for k in ("caption", "description", "result", "output"):
            v = d.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (dict, list, tuple)):
                nested = _extract_caption_text(v)
                if nested:
                    return nested

        # Fallback: first non-empty string-ish value
        for v in d.values():
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""

    # List/tuple outputs
    if isinstance(output, (list, tuple)):
        seq: Sequence[Any] = output
        # Common case: a single dict element
        if len(seq) == 1:
            return _extract_caption_text(seq[0])
        parts: list[str] = []
        for x in seq:
            s = _extract_caption_text(x)
            if s:
                parts.append(s)
        return " ".join(parts).strip()

    # Fallback: stringify and attempt to parse if it looks structured
    as_str = str(output).strip()
    parsed = _maybe_parse_structured_string(as_str)
    if parsed is not as_str:
        return _extract_caption_text(parsed)
    return as_str


def _call_florence2_caption(
    *,
    image_path: Path,
    task_input: str,
    text_input: Optional[str] = None,
) -> str:
    try:
        import replicate  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: replicate. Install it (pip install replicate) and try again."
        ) from e

    with image_path.open("rb") as f:
        model_input: dict[str, object] = {"image": f, "task_input": task_input}
        if text_input:
            model_input["text_input"] = text_input
        output = replicate.run(FLORENCE2_MODEL, input=model_input)

    return _extract_caption_text(output)


def build_caption(*, class_token: str, instance_token: Optional[str], generated: str) -> str:
    """
    Always place class_token at the very start of the caption.
    """
    prefix_parts = [class_token.strip()]
    if instance_token and instance_token.strip():
        prefix_parts.insert(0, instance_token.strip())
    prefix = ", ".join(prefix_parts)

    gen = generated.strip().strip('"').strip()
    if not gen:
        return prefix
    # Avoid duplicating the prefix if the model includes it
    low = gen.lower()
    if low.startswith(class_token.strip().lower()):
        return gen
    return f"{prefix}, {gen}"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing images (captions will be written alongside them).",
    )
    ap.add_argument(
        "--class-token",
        required=True,
        help='Token placed at the top of every caption (first), e.g. "girl" or "ava1 girl".',
    )
    ap.add_argument(
        "--instance-token",
        default=None,
        help='Optional additional token placed after class-token, e.g. "ava1".',
    )
    ap.add_argument(
        "--task-input",
        default="Caption",
        help='Florence-2 task_input, e.g. "Caption", "Detailed Caption", "Object Detection".',
    )
    ap.add_argument(
        "--text-input",
        default=None,
        help="Optional Florence-2 text_input for tasks that require it.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .txt captions if present (default: skip existing).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing files.",
    )

    args = ap.parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"--dataset-dir is not a directory: {dataset_dir}")

    # Load REPLICATE_API_TOKEN from repo-root .env (user will add later)
    repo_root = Path(__file__).resolve().parents[1]
    _load_dotenv_if_present(repo_root / ".env")

    images = _iter_images(dataset_dir)
    if not images:
        print(f"No images found in {dataset_dir} with extensions: {sorted(IMAGE_EXTS)}")
        return 0

    skipped_existing = 0
    planned: list[tuple[Path, Path]] = []
    for img in images:
        cap_path = img.with_suffix(".txt")
        if cap_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue
        planned.append((img, cap_path))

    # Always show a small preview
    print(f"Dataset dir: {dataset_dir}")
    print(f"Found images: {len(images)}")
    print(f"Will write captions: {len(planned)} (skipped existing: {skipped_existing})")
    print(f"Model: {FLORENCE2_MODEL}")
    print(f"Task: {args.task_input}")
    print("")
    print("Planned files (up to 8):")
    for img, cap_path in planned[:8]:
        print(f"- {img.name} -> {cap_path.name}")
    print("")

    if args.dry_run:
        print("Dry-run: no files written.")
        return 0

    if not os.environ.get("REPLICATE_API_TOKEN"):
        raise SystemExit(
            "Missing REPLICATE_API_TOKEN. Add it to environment or put it in /root/kohya-ss-SD-only/.env"
        )

    for img, cap_path in planned:
        print(f"Captioning: {img.name}")
        generated = _call_florence2_caption(
            image_path=img,
            task_input=args.task_input,
            text_input=args.text_input,
        )
        caption = build_caption(
            class_token=args.class_token,
            instance_token=args.instance_token,
            generated=generated,
        )

        tmp = cap_path.with_suffix(cap_path.suffix + ".tmp")
        tmp.write_text(caption.strip() + os.linesep, encoding="utf-8")
        tmp.replace(cap_path)

    print(f"Done. Wrote {len(planned)} caption files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

