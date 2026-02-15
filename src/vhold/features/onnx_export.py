"""ONNX export and INT8 quantization for ProstT5.

One-time conversion from PyTorch to quantized ONNX for faster CPU inference.
Exports the T5 model as three ONNX files (encoder, decoder, decoder_with_past)
and applies dynamic INT8 quantization.

Usage:
    vhold export-onnx
    # Then use: vhold run --backend onnx ...
"""

import platform
import shutil
from pathlib import Path

from vhold.utils.constants import (
    ONNX_MODEL_DIR_NAME,
    PROSTT5_MODEL_NAME,
    get_model_dir,
    get_onnx_model_dir,
)
from vhold.utils.logging import get_logger

logger = get_logger(__name__)

ONNX_FILES = [
    "encoder_model.onnx",
    "decoder_model.onnx",
    "decoder_with_past_model.onnx",
]


def check_onnx_available() -> bool:
    """Check if ONNX Runtime and Optimum are installed."""
    try:
        import onnxruntime  # noqa: F401
        from optimum.onnxruntime import ORTModelForSeq2SeqLM  # noqa: F401

        return True
    except ImportError:
        return False


def check_onnx_exported(model_dir: Path | None = None) -> bool:
    """Check if quantized ONNX models have been exported."""
    onnx_dir = model_dir or get_onnx_model_dir()
    return all((onnx_dir / f).exists() for f in ONNX_FILES)


def _detect_quantization_target() -> str:
    """Auto-detect the best quantization target for this CPU."""
    machine = platform.machine().lower()

    if machine in ("arm64", "aarch64"):
        return "arm64"

    # On x86_64, check for AVX512 VNNI support
    try:
        import cpuinfo

        info = cpuinfo.get_cpu_info()
        flags = info.get("flags", [])
        if "avx512_vnni" in flags:
            return "avx512_vnni"
        if "avx512f" in flags:
            return "avx512"
        return "avx2"
    except (ImportError, Exception):
        # Default to avx2 on x86_64 if cpuinfo unavailable
        return "avx2"


def export_and_quantize(
    model_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    quantization_target: str | None = None,
) -> Path:
    """Export ProstT5 to ONNX and apply INT8 quantization.

    Args:
        model_dir: PyTorch model cache directory (default: ~/.vhold/models)
        output_dir: Output for ONNX models (default: ~/.vhold/models/onnx_int8)
        quantization_target: CPU target (auto-detected if None).
            One of: avx512_vnni, avx512, avx2, arm64

    Returns:
        Path to the output directory containing quantized models
    """
    if not check_onnx_available():
        raise ImportError(
            "ONNX export requires optimum and onnxruntime. "
            "Install with: pip install vhold[onnx]"
        )

    from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    model_cache = Path(model_dir) if model_dir else get_model_dir()
    onnx_output = Path(output_dir) if output_dir else get_onnx_model_dir()

    if quantization_target is None:
        quantization_target = _detect_quantization_target()

    logger.info(f"CPU quantization target: {quantization_target}")

    # Step 1: Export to ONNX (full precision)
    fp32_dir = onnx_output / "_fp32_tmp"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting ProstT5 to ONNX (this requires ~11 GB RAM)...")
    logger.info(f"Model source: {PROSTT5_MODEL_NAME} (cache: {model_cache})")

    model = ORTModelForSeq2SeqLM.from_pretrained(
        PROSTT5_MODEL_NAME,
        export=True,
        cache_dir=model_cache,
    )
    model.save_pretrained(fp32_dir)
    logger.info(f"ONNX export saved to {fp32_dir}")

    # Step 2: Quantize each component
    onnx_output.mkdir(parents=True, exist_ok=True)

    qconfig_fn = {
        "avx512_vnni": lambda: AutoQuantizationConfig.avx512_vnni(
            is_static=False, per_channel=False
        ),
        "avx512": lambda: AutoQuantizationConfig.avx512(
            is_static=False, per_channel=False
        ),
        "avx2": lambda: AutoQuantizationConfig.avx2(
            is_static=False, per_channel=False
        ),
        "arm64": lambda: AutoQuantizationConfig.arm64(
            is_static=False, per_channel=False
        ),
    }

    if quantization_target not in qconfig_fn:
        raise ValueError(
            f"Unknown quantization target: {quantization_target}. "
            f"Valid: {list(qconfig_fn.keys())}"
        )

    dqconfig = qconfig_fn[quantization_target]()
    logger.info(f"Quantizing with {quantization_target} INT8...")

    for onnx_file in ONNX_FILES:
        if not (fp32_dir / onnx_file).exists():
            logger.warning(f"ONNX file not found, skipping: {onnx_file}")
            continue

        logger.info(f"  Quantizing {onnx_file}...")
        quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name=onnx_file)
        quantizer.quantize(save_dir=onnx_output, quantization_config=dqconfig)

    # Step 3: Copy config/tokenizer files
    for config_file in fp32_dir.iterdir():
        if config_file.suffix in (".json", ".model", ".txt") and not (
            onnx_output / config_file.name
        ).exists():
            shutil.copy2(config_file, onnx_output / config_file.name)

    # Step 4: Clean up FP32 intermediates
    shutil.rmtree(fp32_dir)

    # Report sizes
    total_size = sum(
        f.stat().st_size for f in onnx_output.iterdir() if f.suffix == ".onnx"
    )
    logger.info(f"Quantized ONNX models saved to {onnx_output}")
    logger.info(f"Total ONNX size: {total_size / 1e9:.2f} GB")
    logger.info("Use --backend onnx to enable ONNX inference")

    return onnx_output
