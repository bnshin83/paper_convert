from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MarkerPostprocessConfig:
    images_subdir: str = "images"
    strip_spans: bool = True


_SPAN_RE = re.compile(r'<span id="[^"]+"></span>')
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def _is_external_link(path: str) -> bool:
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("data:")
        or path.startswith("file://")
    )


def postprocess_marker_markdown_text(md: str, *, strip_spans: bool) -> str:
    if strip_spans:
        md = _SPAN_RE.sub("", md)
    # Normalize excessive blank lines a bit (keep it conservative)
    md = re.sub(r"\n{4,}", "\n\n\n", md)
    return md


def postprocess_marker_output_dir(
    output_dir: Path, fname_base: str, *, config: MarkerPostprocessConfig = MarkerPostprocessConfig()
) -> dict:
    """
    Marker writes images next to the markdown, e.g.:
      - <out>/<base>.md
      - <out>/_page_3_Figure_0.jpeg

    This postprocess step:
      - moves local image files referenced by markdown into <out>/<images_subdir>/
      - rewrites markdown links to point at images/<file>
      - optionally strips Marker anchor spans (<span id="..."></span>) for cleaner MD
    """
    output_dir = Path(output_dir)
    md_path = output_dir / f"{fname_base}.md"
    if not md_path.exists():
        return {"status": "skipped", "reason": f"missing markdown: {md_path}"}

    md = md_path.read_text(encoding="utf-8", errors="replace")
    md = postprocess_marker_markdown_text(md, strip_spans=config.strip_spans)

    images_dir = output_dir / config.images_subdir
    images_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    rewritten = 0

    def repl(match: re.Match) -> str:
        nonlocal moved, rewritten
        raw_path = match.group(1)
        if _is_external_link(raw_path):
            return match.group(0)

        # Only rewrite relative, local paths that actually exist in this output_dir.
        # (Marker typically writes images next to the .md)
        candidate = (output_dir / raw_path).resolve()
        try:
            candidate.relative_to(output_dir.resolve())
        except Exception:
            return match.group(0)

        if not candidate.exists() or not candidate.is_file():
            return match.group(0)

        dest = images_dir / candidate.name
        if candidate.resolve() != dest.resolve():
            if not dest.exists():
                shutil.move(str(candidate), str(dest))
                moved += 1
            else:
                # Already present; keep the existing one and remove the duplicate if it's different path
                try:
                    candidate.unlink()
                except OSError:
                    pass

        new_path = f"{config.images_subdir}/{dest.name}"
        rewritten += 1
        return match.group(0).replace(raw_path, new_path)

    md2 = _MD_IMAGE_RE.sub(repl, md)
    md_path.write_text(md2, encoding="utf-8")

    return {
        "status": "ok",
        "markdown": str(md_path),
        "images_dir": str(images_dir),
        "images_moved": moved,
        "image_links_rewritten": rewritten,
        "strip_spans": config.strip_spans,
    }


