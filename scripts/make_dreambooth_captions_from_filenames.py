#!/usr/bin/env python3
"""
Create DreamBooth-style caption .txt files next to images.

Example:
  dataset/ava/ava_front_view.png
  -> dataset/ava/ava_front_view.txt containing:
     score_9, score_8_up, score_7_up, rating_safe, source_photo, ava1037, person, asian female, curvy, long wavy brown hair, brown eyes, black bikini, full body, side view, standing, white background, cinematic lighting, smooth skin
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TypedDict


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

DEFAULT_SCORE_TAGS = ("score_9", "score_8_up", "score_7_up")
DEFAULT_RATING_TAG = "rating_safe"
DEFAULT_SOURCE_TAG = "source_photo"
DEFAULT_POSTFIX_TAG = "smooth skin"
DEFAULT_GPT_MODEL = "gpt-5-mini"


class GptTagFields(TypedDict, total=False):
    ethnicity_nationality: list[str]
    body_type: list[str]
    hair_eye: list[str]
    outfit: list[str]
    view_pose: list[str]
    background: list[str]
    lighting_style: list[str]
    extra_tags: list[str]


_GPT_FIELDS_ORDER: tuple[tuple[str, str], ...] = (
    ("ethnicity_nationality", "Ethnicity/Nationality"),
    ("body_type", "Body Type"),
    ("hair_eye", "Hair/Eye Color"),
    ("outfit", "Outfit"),
    ("view_pose", "View/Pose"),
    ("background", "Background"),
    ("lighting_style", "Lighting/Style"),
    ("extra_tags", "Extra"),
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


def _parse_csv_tags(raw: str | None) -> list[str]:
    if raw is None:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _tag_style_normalize(tag: str, *, style: str) -> str:
    t = tag.strip().strip('"').strip("'").strip()
    # Remove trailing punctuation that tends to leak from model outputs
    t = t.strip().strip(".").strip()
    # Never allow commas inside a tag (commas are our separators)
    t = t.replace(",", " ")
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""

    if style == "underscores":
        t = t.replace("/", "_")
        t = t.replace("-", "_")
        t = re.sub(r"\s+", "_", t)
        t = re.sub(r"_+", "_", t).strip("_")
    return t


def _normalize_and_dedupe_tags(tags: Iterable[str], *, style: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        t = _tag_style_normalize(raw, style=style)
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _image_to_data_url(image_path: Path) -> str:
    # Best-effort mime type based on extension (sufficient for OpenAI image_url data URLs).
    ext = image_path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }.get(ext, "application/octet-stream")
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Extract a JSON object from a model response. Prefer the first {...} block.
    """
    s = text.strip()
    if not s:
        return {}
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # Heuristic: take from first '{' to last '}' and try again
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try minimal blocks
    for m in re.finditer(r"\{[\s\S]*?\}", s):
        try:
            return json.loads(m.group(0))
        except Exception:
            continue
    return {}


def _openai_chat_completions(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_data_url: str,
    temperature: Optional[float] = None,
    timeout_s: float = 120.0,
    use_response_format_json: bool = True,
) -> str:
    """
    Call OpenAI Chat Completions with an image. Avoid extra deps (urllib only).
    Returns assistant message content as a string.
    """
    url = "https://api.openai.com/v1/chat/completions"
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        },
    ]

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if use_response_format_json:
        # Some models support strict JSON mode; if unsupported we'll retry without it.
        payload["response_format"] = {"type": "json_object"}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        # Retry once without response_format if it looks like the cause
        if use_response_format_json and ("response_format" in body or "response_format" in str(e)):
            return _openai_chat_completions(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_url=image_data_url,
                temperature=temperature,
                timeout_s=timeout_s,
                use_response_format_json=False,
            )
        # Retry once without temperature if model doesn't support it (e.g. GPT-5 family)
        if temperature is not None and ("temperature" in body and "Only the default (1) value is supported" in body):
            return _openai_chat_completions(
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_url=image_data_url,
                temperature=None,
                timeout_s=timeout_s,
                use_response_format_json=use_response_format_json,
            )
        raise RuntimeError(f"OpenAI API error: HTTP {e.code}. Body: {body[:2000]}") from e
    except Exception as e:
        raise RuntimeError(f"OpenAI API request failed: {e}") from e

    try:
        parsed = json.loads(body)
    except Exception as e:
        raise RuntimeError(f"OpenAI API returned non-JSON: {body[:2000]}") from e

    try:
        return (parsed["choices"][0]["message"]["content"] or "").strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI response shape: {parsed}") from e


def _build_gpt_system_prompt(*, tag_style: str) -> str:
    # We intentionally do not tell the model about instance/class tokens: those are injected locally.
    return (
        "You are generating Danbooru-style comma-separated tags for a single source photo.\n"
        "Return ONLY a single JSON object. No prose. No markdown.\n"
        "\n"
        "Rules:\n"
        "- Output must be JSON with these keys (each value is an array of strings):\n"
        "  ethnicity_nationality, body_type, hair_eye, outfit, view_pose, background, lighting_style, extra_tags\n"
        "- Do NOT include: score tags, rating tags, source tags, instance token, class token, or the postfix tag.\n"
        "- Tags must be short phrases (not full sentences). No commas inside tags.\n"
        "- Prefer visual facts; if uncertain, return an empty list for that field.\n"
        "- Must include hair color and eye color in hair_eye IF visible.\n"
        "- Must include exactly one framing tag inside view_pose: one of\n"
        f'  "face portrait", "upper body", "3/4 body", "full body".\n'
        "- Use lowercase.\n"
        f"- Tag style: {tag_style}.\n"
        "\n"
        "Example output:\n"
        "{\n"
        '  "ethnicity_nationality": ["asian female"],\n'
        '  "body_type": ["curvy"],\n'
        '  "hair_eye": ["long wavy brown hair", "brown eyes"],\n'
        '  "outfit": ["black lingerie"],\n'
        '  "view_pose": ["full body", "side view", "standing"],\n'
        '  "background": ["white background"],\n'
        '  "lighting_style": ["cinematic lighting"],\n'
        '  "extra_tags": []\n'
        "}"
    )


def _build_gpt_user_prompt(*, extra_instruction: str | None) -> str:
    base = (
        "Analyze the image and fill each field with appropriate tags.\n"
        "Keep each tag a short phrase. Avoid redundant synonyms.\n"
    )
    if extra_instruction and extra_instruction.strip():
        base += f"\nAdditional instruction:\n{extra_instruction.strip()}\n"
    return base


def _coerce_to_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Allow comma-separated string fallback
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def _call_gpt_structured_tags(
    *,
    image_path: Path,
    model: str,
    tag_style: str,
    extra_instruction: str | None,
) -> GptTagFields:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY. Add it to environment or put it in /root/kohya-ss-SD-only/.env")

    system_prompt = _build_gpt_system_prompt(tag_style=tag_style)
    user_prompt = _build_gpt_user_prompt(extra_instruction=extra_instruction)
    data_url = _image_to_data_url(image_path)
    raw = _openai_chat_completions(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_data_url=data_url,
        # GPT-5 family models (e.g. gpt-5-mini) do not support custom temperature.
        # Omitting temperature uses the model default (typically 1).
        temperature=None if model.strip().lower().startswith("gpt-5") else 0.2,
    )
    obj = _extract_first_json_object(raw)

    fields: GptTagFields = {}
    for key, _label in _GPT_FIELDS_ORDER:
        fields[key] = _coerce_to_str_list(obj.get(key))
    return fields


def _image_signature(image_path: Path) -> tuple[int, int]:
    st = image_path.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def _load_cache(cache_path: Path) -> dict[str, Any]:
    if not cache_path.exists():
        return {"version": 1, "entries": {}}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "entries" in data and isinstance(data["entries"], dict):
            return data
    except Exception:
        pass
    return {"version": 1, "entries": {}}


def _save_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + os.linesep, encoding="utf-8")
    tmp.replace(cache_path)


def _build_caption_from_template(
    *,
    score_tags: list[str],
    rating_tag: str,
    source_tag: str,
    instance_token: Optional[str],
    fields: GptTagFields,
    postfix_tag: str,
    tag_style: str,
) -> str:
    ordered: list[str] = []
    ordered.extend(score_tags)
    if rating_tag.strip():
        ordered.append(rating_tag.strip())
    if source_tag.strip():
        ordered.append(source_tag.strip())
    if instance_token and instance_token.strip():
        ordered.append(instance_token.strip())

    for key, _label in _GPT_FIELDS_ORDER:
        ordered.extend(fields.get(key, []))

    if postfix_tag.strip():
        ordered.append(postfix_tag.strip())

    final_tags = _normalize_and_dedupe_tags(ordered, style=tag_style)
    return ", ".join(final_tags)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing images (captions will be written alongside them).",
    )
    ap.add_argument(
        "--instance-token",
        required=True,
        help='Trigger word for the character.',
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

    # GPT/template controls
    ap.add_argument(
        "--gpt-model",
        default=DEFAULT_GPT_MODEL,
        help=f'OpenAI model for vision tagging (default: "{DEFAULT_GPT_MODEL}").',
    )
    ap.add_argument(
        "--score-tags",
        default=", ".join(DEFAULT_SCORE_TAGS),
        help='Comma-separated score tags (default: "score_9, score_8_up, score_7_up").',
    )
    ap.add_argument(
        "--rating-tag",
        default=DEFAULT_RATING_TAG,
        help='Rating tag (default: "rating_safe").',
    )
    ap.add_argument(
        "--source-tag",
        default=DEFAULT_SOURCE_TAG,
        help='Source tag (default: "source_photo").',
    )
    ap.add_argument(
        "--postfix-tag",
        default=DEFAULT_POSTFIX_TAG,
        help='Postfix tag (default: "smooth skin").',
    )
    ap.add_argument(
        "--tag-style",
        choices=("spaces", "underscores"),
        default="spaces",
        help='Tag formatting style: "spaces" (e.g. "upper body") or "underscores" (e.g. "upper_body").',
    )
    ap.add_argument(
        "--gpt-extra-instruction",
        default=None,
        help="Optional extra instruction appended to the GPT prompt (use sparingly).",
    )
    ap.add_argument(
        "--cache-path",
        default=None,
        help="Optional cache JSON path for GPT results (default: <dataset-dir>/.caption_cache.json).",
    )

    args = ap.parse_args(argv)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()

    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise SystemExit(f"--dataset-dir is not a directory: {dataset_dir}")

    # Load OPENAI_API_KEY from repo-root .env
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
    print(f"Model: {args.gpt_model}")
    print(f"Score tags: {args.score_tags}")
    print(f"Rating tag: {args.rating_tag}")
    print(f"Source tag: {args.source_tag}")
    print(f"Postfix tag: {args.postfix_tag}")
    print("")
    print("Planned files (up to 8):")
    for img, cap_path in planned[:8]:
        print(f"- {img.name} -> {cap_path.name}")
    print("")

    # Ensure OpenAI key exists early
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Missing OPENAI_API_KEY. Add it to environment or put it in /root/kohya-ss-SD-only/.env"
        )

    cache_path = Path(args.cache_path).expanduser().resolve() if args.cache_path else (dataset_dir / ".caption_cache.json")
    cache = _load_cache(cache_path)
    write_files = not args.dry_run
    if args.dry_run:
        print("Dry-run: captions will be printed; no files written.")

    for img, cap_path in planned:
        print(f"Captioning: {img.name}")
        rel = str(img.relative_to(dataset_dir))
        sig_mtime_ns, sig_size = _image_signature(img)
        entry = (cache.get("entries") or {}).get(rel) if isinstance(cache.get("entries"), dict) else None

        fields: GptTagFields
        if (
            isinstance(entry, dict)
            and entry.get("mtime_ns") == sig_mtime_ns
            and entry.get("size") == sig_size
            and isinstance(entry.get("fields"), dict)
        ):
            # Cache hit
            fields = {k: _coerce_to_str_list(entry["fields"].get(k)) for k, _label in _GPT_FIELDS_ORDER}
        else:
            fields = _call_gpt_structured_tags(
                image_path=img,
                model=args.gpt_model,
                tag_style=args.tag_style,
                extra_instruction=args.gpt_extra_instruction,
            )
            if write_files:
                entries = cache.setdefault("entries", {})
                if isinstance(entries, dict):
                    entries[rel] = {
                        "mtime_ns": sig_mtime_ns,
                        "size": sig_size,
                        "fields": fields,
                    }
                    _save_cache(cache_path, cache)

        score_tags = _parse_csv_tags(args.score_tags)
        caption = _build_caption_from_template(
            score_tags=score_tags or list(DEFAULT_SCORE_TAGS),
            rating_tag=args.rating_tag,
            source_tag=args.source_tag,
            instance_token=args.instance_token,
            fields=fields,
            postfix_tag=args.postfix_tag,
            tag_style=args.tag_style,
        )

        if args.dry_run:
            print(f"- {cap_path.name}: {caption}")
            continue

        tmp = cap_path.with_suffix(cap_path.suffix + ".tmp")
        tmp.write_text(caption.strip() + os.linesep, encoding="utf-8")
        tmp.replace(cap_path)

    print(f"Done. Wrote {len(planned)} caption files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

