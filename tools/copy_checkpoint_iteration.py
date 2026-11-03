"""Copy a checkpoint with new counters, preserving all other state."""

import argparse
import hashlib
from pathlib import Path

import torch


def assert_same_state(expected, actual, path="checkpoint"):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise ValueError(f"Different keys at {path}")
        for key, value in expected.items():
            assert_same_state(value, actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if type(expected) is not type(actual) or len(expected) != len(actual):
            raise ValueError(f"Different sequence at {path}")
        for index, (left, right) in enumerate(zip(expected, actual)):
            assert_same_state(left, right, f"{path}[{index}]")
    elif expected != actual:
        raise ValueError(f"Different value at {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument(
        "--reset-env-step-counter", action="store_true",
        help="Reset the saved environment common_step_counter to zero",
    )
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = args.destination.resolve()
    if args.iteration < 0:
        parser.error("Iteration must be nonnegative")
    if source == destination or destination.exists():
        parser.error("Destination must be a new file; existing files are never overwritten")

    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    previous_iteration = checkpoint["iter"]
    checkpoint["iter"] = args.iteration
    previous_env_steps = None
    if args.reset_env_step_counter:
        infos = checkpoint.get("infos")
        env_state = infos.get("env_state") if isinstance(infos, dict) else None
        if not isinstance(env_state, dict) or "common_step_counter" not in env_state:
            parser.error("Source checkpoint has no saved environment common_step_counter")
        previous_env_steps = env_state["common_step_counter"]
        env_state["common_step_counter"] = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as output:
        torch.save(checkpoint, output)

    copied = torch.load(destination, map_location="cpu", weights_only=True)
    assert_same_state(checkpoint, copied)
    original = torch.load(source, map_location="cpu", weights_only=True)
    copied["iter"] = previous_iteration
    if args.reset_env_step_counter:
        copied["infos"]["env_state"]["common_step_counter"] = previous_env_steps
    assert_same_state(original, copied)
    if hashlib.sha256(source.read_bytes()).hexdigest() != original_hash:
        raise RuntimeError("Source checkpoint changed during verification")
    print(f"Created: {destination}")
    print(f"Iteration: {previous_iteration} -> {args.iteration}")
    if args.reset_env_step_counter:
        print(f"Environment common_step_counter: {previous_env_steps} -> 0")
    print("Verified: source unchanged; all other checkpoint state preserved")


if __name__ == "__main__":
    main()
