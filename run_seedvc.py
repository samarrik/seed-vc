#!/usr/bin/env python3
"""
Standalone Seed-VC runner script.

This script runs in isolation to avoid import conflicts with other ML repos.

Usage:
    python run_seedvc.py --source audio.wav --reference voice.wav --output converted.wav
"""

import argparse
import json
import os
import sys

# Ensure we're using this repo's src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seed_vc_api import ConversionConfig, VoiceConverter


def main():
    parser = argparse.ArgumentParser(
        description="Seed-VC Voice Conversion (Standalone)"
    )
    parser.add_argument("--source", required=True, help="Source audio file")
    parser.add_argument(
        "--reference", required=True, help="Reference voice audio (target voice)"
    )
    parser.add_argument("--output", required=True, help="Output audio path")

    # Conversion config
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7)
    parser.add_argument("--convert-style", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Create converter
    converter = VoiceConverter(
        device=args.device,
        dtype=args.dtype,
        compile_model=args.compile,
        verbose=not args.quiet,
    )

    config = ConversionConfig(
        diffusion_steps=args.diffusion_steps,
        length_adjust=args.length_adjust,
        intelligibility_cfg_rate=args.intelligibility_cfg_rate,
        similarity_cfg_rate=args.similarity_cfg_rate,
        convert_style=args.convert_style,
        seed=args.seed,
    )

    output_path = converter.convert(
        source=args.source,
        reference=args.reference,
        output=args.output,
        config=config,
    )

    # Output result as JSON for parsing
    result = {"success": True, "output": output_path}
    print(f"\n__RESULT_JSON__:{json.dumps(result)}")


if __name__ == "__main__":
    main()
