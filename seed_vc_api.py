"""
Seed-VC Voice Conversion API

Zero-shot voice conversion with in-context learning.

Features:
    - NO LENGTH LIMIT on source audio (processed in streaming chunks)
    - Reference audio: 1-25 seconds for voice cloning
    - Supports any audio format (wav, mp3, flac, etc.)

Args:
    --source: Path to source audio file (any length)
    --reference: Path to reference audio file (target voice, 1-25s)
    --output: Path to output audio file

Example:
    from seed_vc_api import VoiceConverter
    
    converter = VoiceConverter()
    converter.convert(
        source="source.wav",      # Can be hours long!
        reference="reference.wav", # Short voice sample
        output="output.wav"
    )
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import yaml
import soundfile as sf


@dataclass
class ConversionConfig:
    """
    Voice conversion configuration with sensible defaults.
    
    Note on audio lengths:
        - Source audio: NO LENGTH LIMIT - processed in chunks (streaming)
        - Reference audio: Used for voice cloning, 1-25 seconds recommended
    """
    
    diffusion_steps: int = 30
    """Number of diffusion steps (30-50 for quality, 4-10 for speed)."""
    
    length_adjust: float = 1.0
    """Length adjustment factor (<1.0 faster, >1.0 slower)."""
    
    intelligibility_cfg_rate: float = 0.7
    """Clarity of output speech (0.0-1.0)."""
    
    similarity_cfg_rate: float = 0.7
    """Similarity to reference voice (0.0-1.0)."""
    
    convert_style: bool = False
    """Enable accent/emotion conversion (uses AR model)."""
    
    anonymization_only: bool = False
    """Anonymize to average voice, ignoring reference."""
    
    top_p: float = 0.9
    """AR model diversity (0.5-1.0)."""
    
    temperature: float = 1.0
    """AR model randomness (0.7-1.2)."""
    
    repetition_penalty: float = 1.0
    """Repetition penalty for AR model (1.0-1.5)."""
    
    seed: int = 42
    """Random seed for reproducibility."""


class VoiceConverter:
    """
    Voice conversion model wrapper.
    
    Loads models once on initialization, can convert multiple times.
    
    Args:
        device: Device to use ("cuda", "cpu", or "mps")
        dtype: Data type ("fp16" or "fp32")
        compile_model: Use torch.compile for faster inference
        ar_checkpoint: Path to custom AR checkpoint (optional)
        cfm_checkpoint: Path to custom CFM checkpoint (optional)
        verbose: Print progress messages
    """
    
    def __init__(
        self,
        device: str = "cuda",
        dtype: str = "fp16",
        compile_model: bool = False,
        ar_checkpoint: Optional[str] = None,
        cfm_checkpoint: Optional[str] = None,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self._log("Initializing VoiceConverter...")
        
        # Setup device
        if device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif device == "mps" and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        self.dtype = torch.float16 if dtype == "fp16" else torch.float32
        self._log(f"Using device: {self.device}, dtype: {self.dtype}")
        
        # CUDA optimizations
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        
        # Load model
        self._load_model(ar_checkpoint, cfm_checkpoint, compile_model)
        self._log("VoiceConverter ready!")
    
    def _log(self, msg: str):
        """Print message if verbose mode enabled."""
        if self.verbose:
            print(f"[VoiceConverter] {msg}")
    
    def _load_model(
        self,
        ar_checkpoint: Optional[str],
        cfm_checkpoint: Optional[str],
        compile_model: bool,
    ):
        """Load the voice conversion model."""
        from hydra.utils import instantiate
        from omegaconf import DictConfig
        
        config_path = os.path.join(os.path.dirname(__file__), "configs", "vc_wrapper.yaml")
        cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
        
        self._log("Loading model components...")
        self.model = instantiate(cfg)
        
        self._log("Loading checkpoints...")
        self.model.load_checkpoints(
            ar_checkpoint_path=ar_checkpoint,
            cfm_checkpoint_path=cfm_checkpoint,
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Setup KV caches for AR model
        self.model.setup_ar_caches(
            max_batch_size=1,
            max_seq_len=4096,
            dtype=self.dtype,
            device=self.device,
        )
        
        # Optional compilation
        if compile_model:
            self._log("Compiling model with torch.compile...")
            torch._inductor.config.coordinate_descent_tuning = True
            torch._inductor.config.triton.unique_kernel_names = True
            if hasattr(torch._inductor.config, "fx_graph_cache"):
                torch._inductor.config.fx_graph_cache = True
            self.model.compile_ar()
    
    def convert(
        self,
        source: str,
        reference: str,
        output: str,
        config: Optional[ConversionConfig] = None,
    ) -> str:
        """
        Convert voice from source to match reference.
        
        Supports unlimited source audio length - processed in chunks automatically.
        Reference audio is used for voice cloning (1-25 seconds recommended).
        
        Args:
            source: Path to source audio file (any length supported)
            reference: Path to reference audio file (1-25s for voice sample)
            output: Path to save converted audio
            config: Conversion configuration (uses defaults if None)
        
        Returns:
            Path to the output file
        """
        config = config or ConversionConfig()
        
        # Validate inputs
        if not os.path.exists(source):
            raise FileNotFoundError(f"Source file not found: {source}")
        if not os.path.exists(reference):
            raise FileNotFoundError(f"Reference file not found: {reference}")
        
        # Create output directory
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        
        # Set seed
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(config.seed)
        
        self._log(f"Converting: {source} -> {output}")
        self._log(f"Reference: {reference}")
        
        start_time = time.time()
        
        # Run conversion
        generator = self.model.convert_voice_with_streaming(
            source_audio_path=source,
            target_audio_path=reference,
            diffusion_steps=config.diffusion_steps,
            length_adjust=config.length_adjust,
            intelligebility_cfg_rate=config.intelligibility_cfg_rate,
            similarity_cfg_rate=config.similarity_cfg_rate,
            top_p=config.top_p,
            temperature=config.temperature,
            repetition_penalty=config.repetition_penalty,
            convert_style=config.convert_style,
            anonymization_only=config.anonymization_only,
            device=self.device,
            dtype=self.dtype,
            stream_output=True,
        )
        
        # Collect output
        full_audio = None
        for _, full_audio in generator:
            pass
        
        if full_audio is None:
            raise RuntimeError("Voice conversion failed - no output generated")
        
        # Save output
        sample_rate, audio_data = full_audio
        sf.write(output, audio_data, sample_rate)
        
        elapsed = time.time() - start_time
        self._log(f"Conversion complete in {elapsed:.2f}s")
        self._log(f"Output saved to: {output}")
        
        return output


def _str2bool(v):
    """Convert string to boolean for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed-VC Voice Conversion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required arguments
    parser.add_argument("--source", required=True, help="Source audio file")
    parser.add_argument("--reference", required=True, help="Reference voice audio")
    parser.add_argument("--output", required=True, help="Output audio file")
    
    # Model options
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--compile", type=_str2bool, default=False, help="Use torch.compile")
    parser.add_argument("--ar-checkpoint", default=None, help="Custom AR checkpoint")
    parser.add_argument("--cfm-checkpoint", default=None, help="Custom CFM checkpoint")
    
    # Conversion options
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    parser.add_argument("--convert-style", type=_str2bool, default=False)
    parser.add_argument("--anonymization-only", type=_str2bool, default=False)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    args = parser.parse_args()
    
    # Create config
    config = ConversionConfig(
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligibility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        convert_style=args.convert_style,
        anonymization_only=args.anonymization_only,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    
    # Run conversion
    converter = VoiceConverter(
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        ar_checkpoint=args.ar_checkpoint,
        cfm_checkpoint=args.cfm_checkpoint,
        verbose=not args.quiet,
    )
    
    converter.convert(
        source=args.source,
        reference=args.reference,
        output=args.output,
        config=config,
    )
