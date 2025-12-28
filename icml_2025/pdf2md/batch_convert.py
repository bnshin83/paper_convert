#!/usr/bin/env python3
"""
Batch PDF to Markdown Converter
Supports: MinerU (best accuracy) and Marker (fastest)

Usage:
    python batch_convert.py --input_dir ./pdfs --output_dir ./markdown --engine mineru
    python batch_convert.py --input_dir ./pdfs --output_dir ./markdown --engine marker --workers 4
"""

import argparse
import os
import sys
import time
import json
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Local helper to make Marker output more "Mathpix-like"
    from postprocess_marker import MarkerPostprocessConfig, postprocess_marker_output_dir
except Exception:  # pragma: no cover
    MarkerPostprocessConfig = None  # type: ignore
    postprocess_marker_output_dir = None  # type: ignore


def convert_with_mineru(pdf_path: Path, output_dir: Path, use_gpu: bool = True) -> dict:
    """Convert PDF using MinerU (magic-pdf)."""
    try:
        from magic_pdf.pipe.UNIPipe import UNIPipe
        from magic_pdf.rw.DiskReaderWriter import DiskReaderWriter
        import magic_pdf.model as model_config

        model_config.__use_inside_model__ = True

        pdf_bytes = pdf_path.read_bytes()
        output_subdir = output_dir / pdf_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        image_writer = DiskReaderWriter(str(output_subdir / "images"))

        pipe = UNIPipe(pdf_bytes, {"_pdf_type": "", "model_list": []}, image_writer)
        pipe.pipe_classify()
        pipe.pipe_analyze()
        pipe.pipe_parse()

        md_content = pipe.pipe_mk_markdown(str(output_subdir / "images"), drop_mode="none")

        md_path = output_subdir / f"{pdf_path.stem}.md"
        md_path.write_text(md_content, encoding='utf-8')

        return {"status": "success", "pdf": str(pdf_path), "output": str(md_path)}

    except Exception as e:
        return {"status": "error", "pdf": str(pdf_path), "error": str(e)}


def convert_with_mineru_cli(pdf_path: Path, output_dir: Path) -> dict:
    """Convert PDF using MinerU CLI (recommended for batch)."""
    try:
        output_subdir = output_dir / pdf_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "magic-pdf",
            "-p", str(pdf_path),
            "-o", str(output_subdir),
            "-m", "auto"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Log stderr for debugging
        if result.stderr:
            logger.warning(f"magic-pdf stderr for {pdf_path.name}: {result.stderr[-500:]}")

        if result.returncode == 0:
            # Find the generated markdown file
            md_files = list(output_subdir.rglob("*.md"))
            if md_files:
                return {"status": "success", "pdf": str(pdf_path), "output": str(md_files[0])}
            # No markdown files found - this is actually an error
            return {"status": "error", "pdf": str(pdf_path), "error": f"No .md files generated. stderr: {result.stderr[-200:] if result.stderr else 'none'}"}
        else:
            return {"status": "error", "pdf": str(pdf_path), "error": result.stderr}

    except subprocess.TimeoutExpired:
        return {"status": "error", "pdf": str(pdf_path), "error": "Timeout (300s)"}
    except Exception as e:
        return {"status": "error", "pdf": str(pdf_path), "error": str(e)}


def convert_with_marker(
    pdf_path: Path,
    output_dir: Path,
    use_llm: bool = False,
    *,
    marker_images_subdir: str = "images",
    marker_strip_spans: bool = True,
) -> dict:
    """Convert PDF using Marker."""
    try:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import save_output

        output_subdir = output_dir / pdf_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        models = create_model_dict()
        converter = PdfConverter(artifact_dict=models)
        rendered = converter(str(pdf_path))
        # Writes: <stem>.md + <stem>_meta.json + extracted images (figures) referenced by the markdown
        save_output(rendered, str(output_subdir), pdf_path.stem)
        md_path = output_subdir / f"{pdf_path.stem}.md"

        if postprocess_marker_output_dir is not None:
            postprocess_marker_output_dir(
                output_subdir,
                pdf_path.stem,
                config=MarkerPostprocessConfig(
                    images_subdir=marker_images_subdir,
                    strip_spans=marker_strip_spans,
                ),
            )

        return {"status": "success", "pdf": str(pdf_path), "output": str(md_path)}

    except Exception as e:
        return {"status": "error", "pdf": str(pdf_path), "error": str(e)}


def convert_with_marker_cli(
    pdf_path: Path,
    output_dir: Path,
    use_llm: bool = False,
    *,
    marker_images_subdir: str = "images",
    marker_strip_spans: bool = True,
) -> dict:
    """Convert PDF using Marker CLI."""
    try:
        cmd = ["marker_single", str(pdf_path), "--output_dir", str(output_dir)]
        if use_llm:
            cmd.append("--use_llm")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            output_subdir = output_dir / pdf_path.stem
            md_files = list(output_subdir.rglob("*.md")) if output_subdir.exists() else []
            if md_files:
                if postprocess_marker_output_dir is not None:
                    postprocess_marker_output_dir(
                        output_subdir,
                        pdf_path.stem,
                        config=MarkerPostprocessConfig(
                            images_subdir=marker_images_subdir,
                            strip_spans=marker_strip_spans,
                        ),
                    )
                return {"status": "success", "pdf": str(pdf_path), "output": str(md_files[0])}
            return {"status": "success", "pdf": str(pdf_path), "output": str(output_dir)}
        else:
            return {"status": "error", "pdf": str(pdf_path), "error": result.stderr}

    except subprocess.TimeoutExpired:
        return {"status": "error", "pdf": str(pdf_path), "error": "Timeout (300s)"}
    except Exception as e:
        return {"status": "error", "pdf": str(pdf_path), "error": str(e)}


def batch_convert_marker(
    input_dir: Path,
    output_dir: Path,
    use_llm: bool = False,
    *,
    marker_images_subdir: str = "images",
    marker_strip_spans: bool = True,
) -> dict:
    """Use Marker's native batch conversion (fastest on GPU)."""
    try:
        cmd = [
            "marker",
            str(input_dir),
            "--output_dir", str(output_dir),
            "--workers", "1",  # Per-GPU workers
        ]
        if use_llm:
            cmd.append("--use_llm")

        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=False, text=True)

        # Postprocess outputs to make markdown cleaner + consolidate images.
        if result.returncode == 0 and postprocess_marker_output_dir is not None:
            for md_path in output_dir.rglob("*.md"):
                # Assume md is at <out>/<stem>/<stem>.md (standard in this repo)
                out_subdir = md_path.parent
                stem = md_path.stem
                postprocess_marker_output_dir(
                    out_subdir,
                    stem,
                    config=MarkerPostprocessConfig(
                        images_subdir=marker_images_subdir,
                        strip_spans=marker_strip_spans,
                    ),
                )

        return {"status": "success" if result.returncode == 0 else "error"}

    except Exception as e:
        return {"status": "error", "error": str(e)}


def process_single_pdf(args: tuple) -> dict:
    """Process a single PDF (for parallel processing)."""
    pdf_path, output_dir, engine, use_llm, marker_images_subdir, marker_strip_spans = args
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if engine == "mineru":
        return convert_with_mineru_cli(pdf_path, output_dir)
    elif engine == "marker":
        return convert_with_marker_cli(
            pdf_path,
            output_dir,
            use_llm,
            marker_images_subdir=marker_images_subdir,
            marker_strip_spans=marker_strip_spans,
        )
    else:
        return {"status": "error", "pdf": str(pdf_path), "error": f"Unknown engine: {engine}"}


def main():
    parser = argparse.ArgumentParser(description="Batch PDF to Markdown Converter")
    parser.add_argument("--input_dir", "-i", type=str, required=True, help="Input directory with PDFs")
    parser.add_argument("--output_dir", "-o", type=str, required=True, help="Output directory for markdown")
    parser.add_argument("--engine", "-e", type=str, default="mineru", choices=["mineru", "marker"],
                        help="Conversion engine (default: mineru)")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--use_llm", action="store_true", help="Use LLM enhancement (Marker only)")
    parser.add_argument("--batch_mode", action="store_true", help="Use native batch mode (Marker only)")
    parser.add_argument("--resume", action="store_true", help="Skip already converted PDFs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of PDFs to process")
    parser.add_argument("--images_subdir", type=str, default="images",
                        help="For Marker outputs: move extracted figures into this subfolder and rewrite links (default: images)")
    parser.add_argument("--keep_spans", action="store_true",
                        help="Keep Marker's <span id=...> anchors in markdown (default: strip them)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all PDFs
    pdf_files = list(input_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDFs in {input_dir}")

    # Resume support - skip already converted
    if args.resume:
        existing = set()
        for md_file in output_dir.rglob("*.md"):
            existing.add(md_file.stem)
        pdf_files = [p for p in pdf_files if p.stem not in existing]
        logger.info(f"Resuming: {len(pdf_files)} PDFs remaining")

    # Apply limit
    if args.limit:
        pdf_files = pdf_files[:args.limit]
        logger.info(f"Limited to {len(pdf_files)} PDFs")

    if not pdf_files:
        logger.info("No PDFs to process")
        return

    # Use native batch mode for Marker (fastest)
    if args.batch_mode and args.engine == "marker":
        logger.info("Using Marker native batch mode")
        result = batch_convert_marker(
            input_dir,
            output_dir,
            args.use_llm,
            marker_images_subdir=args.images_subdir,
            marker_strip_spans=not args.keep_spans,
        )
        logger.info(f"Batch result: {result}")
        return

    # Parallel processing
    start_time = time.time()
    results = {"success": 0, "error": 0, "errors": []}

    work_items = [
        (str(p), str(output_dir), args.engine, args.use_llm, args.images_subdir, not args.keep_spans)
        for p in pdf_files
    ]

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_single_pdf, item): item[0] for item in work_items}

            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["status"] == "success":
                    results["success"] += 1
                else:
                    results["error"] += 1
                    results["errors"].append(result)

                if (i + 1) % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    logger.info(f"Progress: {i+1}/{len(pdf_files)} ({rate:.2f} PDFs/sec)")
    else:
        for i, item in enumerate(work_items):
            result = process_single_pdf(item)
            if result["status"] == "success":
                results["success"] += 1
            else:
                results["error"] += 1
                results["errors"].append(result)

            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                logger.info(f"Progress: {i+1}/{len(pdf_files)} ({rate:.2f} PDFs/sec)")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*50}")
    logger.info(f"Completed in {elapsed:.2f}s")
    logger.info(f"Success: {results['success']}, Errors: {results['error']}")
    logger.info(f"Rate: {len(pdf_files)/elapsed:.2f} PDFs/sec")

    # Save error log
    if results["errors"]:
        error_log = output_dir / "errors.json"
        error_log.write_text(json.dumps(results["errors"], indent=2))
        logger.info(f"Error log saved to {error_log}")


if __name__ == "__main__":
    main()
