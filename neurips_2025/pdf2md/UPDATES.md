# pdf2md updates (Dec 2025)

This file summarizes the changes made in this workspace to improve stability on SLURM + GPU and to improve Markdown output quality (figures + cleaner MD).

## GPU / SLURM stability

- **Problem observed**: On some nodes (notably `h000`), PyTorch/Marker could see the GPU but CUDA context creation failed with:
  - `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- **Outcome**: The same workload ran fine on other nodes (e.g. `h003`), suggesting a **node-specific / GPU health** issue rather than a code issue.

### `PDF2MD_DEVICE` control (GPU-only by default)

Updated `slurm_test.sh` to support:
- **`PDF2MD_DEVICE=cuda`** (default): **GPU-only**, fail fast if CUDA allocation fails.
- **`PDF2MD_DEVICE=auto`**: use GPU if healthy, otherwise fall back to CPU.
- **`PDF2MD_DEVICE=cpu`**: force CPU.

Recommended GPU-only submit (avoid known-bad node):

```bash
cd /scratch/gautschi/shin283/paper/neurips_2025/pdf2md
sbatch --exclude=h000 --export=ALL,PDF2MD_DEVICE=cuda slurm_test.sh
```

## Figures (images) in Markdown

### What was wrong

Marker returns extracted figures as an in-memory dict (`rendered.images`), and the Markdown references local filenames.  
We were previously writing only the `.md` text, so **figures were not saved** to disk.

### What was changed

- `slurm_test.sh` now uses `marker.output.save_output(...)` so each PDF output directory contains:
  - `<paper>.md`
  - `<paper>_meta.json`
  - extracted figure/picture images (`*.jpeg`/`*.png`)
- `batch_convert.py` was updated similarly for the Marker Python API path.

## “SOTA-ish” Markdown cleanup (more Mathpix-like)

Mathpix-style output tends to be:
- clean Markdown
- figures referenced as image links
- minimal extra HTML noise

To move toward that, we added a post-processing step for Marker outputs:

### New: `postprocess_marker.py`

File: `postprocess_marker.py`

It performs:
- **Image consolidation**: move extracted images into an `images/` subfolder
- **Link rewriting**: rewrite markdown image links to `images/<file>`
- **Cleaner MD**: strip Marker anchor spans like `<span id="page-..."></span>` by default

Example after postprocess:
- Before: `![](_page_3_Figure_0.jpeg)`
- After: `![](images/_page_3_Figure_0.jpeg)`

### Wired into all Marker execution paths

`batch_convert.py` now runs this postprocess for:
- Marker CLI single (`marker_single`)
- Marker CLI batch (`marker`)
- Marker Python API conversion

`slurm_test.sh` also runs the postprocess so the sample outputs look clean by default.

## New CLI flags (batch_convert.py)

When using `--engine marker`:
- **`--images_subdir <name>`**: where to move images (default: `images`)
- **`--keep_spans`**: keep Marker `<span id=...>` anchors (default: strip them)

Example:

```bash
python batch_convert.py -i ./pdfs -o ./markdown -e marker -w 4 --images_subdir images
```

## Files changed / added

- **Modified**:
  - `paper/neurips_2025/pdf2md/slurm_test.sh`
  - `paper/neurips_2025/pdf2md/batch_convert.py`
- **Added**:
  - `paper/neurips_2025/pdf2md/postprocess_marker.py`
  - `paper/neurips_2025/pdf2md/UPDATES.md` (this file)


