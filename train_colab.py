"""Train a metal-nut PatchCore model on MVTec AD and export it to ONNX.

Run in Colab or another environment with anomalib and PyTorch installed:
    python train_colab.py --data ./datasets/MVTecAD
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("./datasets/MVTecAD"))
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "backend" / "model.onnx")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    try:
        from anomalib.data import MVTec
        from anomalib.models import Patchcore
        from anomalib.engine import Engine
    except ImportError as exc:
        raise SystemExit("Install anomalib and torch first: pip install anomalib torch") from exc

    # MVTec's training split contains only normal (train/good) samples. Test
    # anomalies are intentionally never used to fit the memory bank.
    datamodule = MVTec(root=str(args.data), category="metal_nut")
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )
    engine = Engine(max_epochs=args.epochs)
    engine.fit(model=model, datamodule=datamodule)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    example = torch.zeros(1, 3, 256, 256, device=device)
    # Export the trained inference graph. Dynamic batch/size axes support both
    # uploaded images and the fixed-size live camera frames used by the app.
    torch.onnx.export(
        model,
        example,
        str(args.output),
        input_names=["image"],
        output_names=["prediction"],
        dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"}, "prediction": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"Exported trained PatchCore model to {args.output}")


if __name__ == "__main__":
    main()
