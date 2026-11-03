"""Export a current point-foot actor, including its native inference normalization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

if __package__:
    from .pointleg_sim2sim import load_policy
else:
    from pointleg_sim2sim import load_policy

from pointleg import TASK_SPECS


def export_policy(checkpoint: Path, task: str, output: Path | None = None) -> tuple[Path, float]:
    """Publish an ONNX file only after native-actor and runtime outputs agree."""
    checkpoint = checkpoint.expanduser().resolve()
    if checkpoint.suffix.lower() not in (".pt", ".pth"):
        raise ValueError("Input must be a point-foot MJLab .pt or .pth checkpoint")
    output = output.expanduser().resolve() if output is not None else checkpoint.with_suffix(".onnx")
    if output.suffix.lower() != ".onnx":
        raise ValueError("Output must have the .onnx extension")
    model = load_policy(checkpoint, task)
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Export verification requires onnx and onnxruntime; use sim2sim/requirements.txt") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(42)
    max_diff = 0.0
    with tempfile.TemporaryDirectory(prefix="pointleg-export-", dir=output.parent) as directory:
        candidate = Path(directory) / "policy.onnx"
        torch.onnx.export(
            model, torch.zeros(1, 45), str(candidate),
            export_params=True, opset_version=14, dynamo=False,
            input_names=["obs"], output_names=["actions"],
            dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
        )
        if task == "PointLeg-Jump-v0":
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            exported = onnx.load(str(candidate))
            onnx.helper.set_model_props(exported, {
                "pointleg_jump_contract": json.dumps(saved["infos"]["jump_contract"], default=str),
            })
            onnx.save(exported, str(candidate))
        onnx.checker.check_model(str(candidate))
        session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
        samples = [torch.zeros(1, 45), torch.randn(1, 45, generator=generator),
                   torch.randn(4, 45, generator=generator)]
        for obs in samples:
            with torch.inference_mode():
                expected = model(obs).numpy()
            actual = session.run(["actions"], {"obs": obs.numpy()})[0]
            if actual.shape != (obs.shape[0], 12) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
                raise ValueError("Export must produce 12 finite actions per observation")
            try:
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            except AssertionError as exc:
                raise ValueError("ONNX verification failed: output differs from the native PyTorch actor") from exc
            max_diff = max(max_diff, float(np.max(np.abs(actual - expected))))
        del session
        os.replace(candidate, output)
    return output, max_diff


def main():
    parser = argparse.ArgumentParser(description="Export a 45-observation, 12-action point-foot policy to ONNX")
    parser.add_argument("--task", choices=[name for name, spec in TASK_SPECS.items() if spec.enabled], required=True)
    parser.add_argument(
        "--checkpoint", "--pt-path", "--pt_path", dest="checkpoint", type=Path, required=True,
        help="Point-foot checkpoint matching the selected task's current network configuration",
    )
    parser.add_argument("--output", type=Path, help="Output .onnx path; default is beside the checkpoint")
    args = parser.parse_args()
    try:
        output, max_diff = export_policy(args.checkpoint, args.task, args.output)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Export error: {exc}\n")
    print(f"Exported and verified: {output}")
    print(f"Task: {args.task}; observations: 45; actions: 12; max absolute error: {max_diff:.8g}")


if __name__ == "__main__":
    main()
