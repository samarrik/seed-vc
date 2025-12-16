# Seed-VC

Zero-shot voice conversion with in-context learning. Clone any voice with 1-25 seconds of reference audio.

**No length limit on source audio** - convert hours of audio in a single call!

## Setup

```bash
# Create environment
conda create -n seed-vc python=3.10
conda activate seed-vc

# Install dependencies
pip install -r requirements.txt
```

## Download Models

Models are downloaded automatically from HuggingFace on first run.

> **Note**: If HuggingFace is inaccessible, set `HF_ENDPOINT=https://hf-mirror.com` before running.

## Usage

### Python API

```python
from seed_vc_api import VoiceConverter

converter = VoiceConverter()
converter.convert(
    source="source.wav",
    reference="reference.wav",
    output="output.wav"
)
```

### CLI

```bash
python seed_vc_api.py \
    --source source.wav \
    --reference reference.wav \
    --output output.wav
```

### Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--source` | Yes | Source audio file - **any length supported** |
| `--reference` | Yes | Reference voice sample (1-25 seconds) |
| `--output` | Yes | Output audio file path |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--device` | `cuda` | Device: `cuda`, `cpu`, or `mps` |
| `--dtype` | `fp16` | Data type: `fp16` or `fp32` |
| `--diffusion-steps` | `30` | Quality vs speed (30-50 quality, 4-10 fast) |
| `--length-adjust` | `1.0` | Speed factor (<1.0 faster, >1.0 slower) |
| `--intelligibility-cfg-rate` | `0.7` | Speech clarity (0.0-1.0) |
| `--similarity-cfg-rate` | `0.7` | Voice similarity (0.0-1.0) |
| `--convert-style` | `false` | Enable accent/emotion conversion |
| `--anonymization-only` | `false` | Anonymize to average voice |
| `--compile` | `false` | Use torch.compile for speed |
| `--quiet` | - | Suppress progress output |

## Apptainer (Cluster)

### Build

```bash
apptainer build seed_vc.sif seed_vc.def
```

### Run

```bash
# Load modules (cluster-specific)
module load CUDA/12.2.0
module load FFmpeg

# Run conversion
apptainer run --nv \
    --bind /path/to/weights:/app/pretrained_weights \
    --bind /path/to/data:/data \
    seed_vc.sif \
    --source /data/source.wav \
    --reference /data/reference.wav \
    --output /data/output.wav
```

## Project Structure

```
seed-vc/
├── seed_vc_api.py      # Main API
├── seed_vc.def         # Apptainer definition
├── requirements.txt    # Dependencies
├── configs/
│   └── vc_wrapper.yaml
└── src/
    ├── hf_utils.py     # HuggingFace utilities
    ├── commons.py      # Common utilities
    ├── audio.py        # Audio processing
    ├── v2/             # Core model
    ├── astral_quantization/
    ├── campplus/
    └── bigvgan/
```

## License

See [LICENSE](LICENSE) file.
