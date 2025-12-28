# pdf2md

Open-source batch PDF to Markdown converter for academic papers. A local alternative to Mathpix with comparable quality.

## Features

- **Two engines**: MinerU (best accuracy) and Marker (fastest)
- **Academic-focused**: Handles LaTeX math, tables, figures, citations
- **Batch processing**: Convert thousands of PDFs in parallel
- **GPU accelerated**: Optimized for NVIDIA GPUs (H100, A100, etc.)
- **Apple Silicon**: Native MPS support for M1/M2/M3/M4 Macs
- **Resume support**: Continue interrupted conversions
- **SLURM ready**: Included job scripts for HPC clusters

## Performance

| Engine | Speed (H100) | Accuracy | Best For |
|--------|--------------|----------|----------|
| **MinerU** | ~5-10 pages/sec | ★★★★★ | Academic papers, math-heavy |
| **Marker** | ~25 pages/sec | ★★★★☆ | Fast batch processing |

## Installation

### Quick Install

```bash
# Clone the repo
git clone https://github.com/bnshin83/pdf2md.git
cd pdf2md

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with MinerU (recommended for academic papers)
pip install -e ".[mineru]"

# Or install with Marker (faster)
pip install -e ".[marker]"

# Or install both
pip install -e ".[all]"
```

### HPC Cluster (SLURM)

```bash
# Clone on cluster
git clone https://github.com/bnshin83/pdf2md.git
cd pdf2md

# Run setup script
chmod +x setup_cluster.sh
./setup_cluster.sh
```

### Apple Silicon Mac

```bash
git clone https://github.com/bnshin83/pdf2md.git
cd pdf2md
chmod +x setup_local.sh
./setup_local.sh
```

## Usage

### Command Line

```bash
# Activate environment
source .venv/bin/activate

# Single PDF with MinerU
magic-pdf -p paper.pdf -o output/

# Single PDF with Marker
marker_single paper.pdf --output_dir output/

# Batch convert directory
python batch_convert.py -i ./pdfs -o ./markdown -e mineru -w 4

# Resume interrupted conversion
python batch_convert.py -i ./pdfs -o ./markdown -e mineru --resume

# Marker native batch (fastest)
marker ./pdfs --output_dir ./markdown
```

### SLURM Job

```bash
# Edit paths in slurm_batch_convert.sh, then:
sbatch slurm_batch_convert.sh

# Monitor
squeue -u $USER
tail -f logs/*.out
```

### Python API

```python
from pathlib import Path
from batch_convert import convert_with_mineru_cli, convert_with_marker_cli

# Single PDF
result = convert_with_mineru_cli(Path("paper.pdf"), Path("output/"))
print(result)  # {"status": "success", "output": "output/paper.md"}

# Or with Marker
result = convert_with_marker_cli(Path("paper.pdf"), Path("output/"))
```

## Output Structure

```
output/
├── paper1/
│   ├── paper1.md          # Markdown with LaTeX math
│   └── images/
│       ├── fig1.png
│       └── table1.png
├── paper2/
│   └── paper2.md
└── errors.json            # Failed conversions (if any)
```

## Configuration

### MinerU Config (`~/.magic-pdf/magic-pdf.json`)

```json
{
  "device": "cuda",
  "table-engine": "rapidocr",
  "formula-engine": "rapidocr"
}
```

For Apple Silicon, use `"device": "mps"`.

### Marker with LLM Enhancement

```bash
# Set API key for Gemini
export GOOGLE_API_KEY=your_key

# Run with LLM (better table/math formatting)
python batch_convert.py -i ./pdfs -o ./markdown -e marker --use_llm
```

## Benchmarks

Tested on NeurIPS 2025 papers (~5000 PDFs, ~10 pages each):

| Setup | Engine | Time | Success Rate |
|-------|--------|------|--------------|
| H100 (1 GPU) | Marker | ~45 min | 98.5% |
| H100 (1 GPU) | MinerU | ~2.5 hr | 99.2% |
| M4 Max | Marker | ~4 hr | 98.0% |
| M4 Max | MinerU | ~8 hr | 99.0% |

## Troubleshooting

### CUDA Out of Memory

```bash
# Reduce workers
python batch_convert.py -i ./pdfs -o ./markdown -e mineru -w 1
```

### Models Not Found

```python
# Re-download MinerU models
from huggingface_hub import snapshot_download
snapshot_download(repo_id='opendatalab/PDF-Extract-Kit')

# Re-download Marker models
from marker.models import create_model_dict
models = create_model_dict()
```

### Slow on Mac

Ensure MPS is enabled:
```bash
cat ~/.magic-pdf/magic-pdf.json
# Should show: {"device": "mps"}
```

## License

MIT

## Acknowledgments

- [MinerU](https://github.com/opendatalab/MinerU) by OpenDataLab
- [Marker](https://github.com/VikParuchuri/marker) by Vik Paruchuri
