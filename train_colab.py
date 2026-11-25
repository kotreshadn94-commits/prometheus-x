"""Train PatchCore on MVTec AD metal_nut train/good and export an ONNX graph.

Run from the repository root in an environment with anomalib, PyTorch and ONNX installed.
"""
from pathlib import Path
import torch
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "backend" / "model.onnx"


def main() -> None:
    datamodule = MVTec(root=str(ROOT / "datasets" / "MVTec"), category="metal_nut")
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"], coreset_sampling_ratio=0.1)
    engine = Engine()
    engine.fit(model=model, datamodule=datamodule)
    model.eval()
    # Export an image-to-prediction graph. Some anomalib releases expose a Lightning
    # wrapper with a structured output; the wrapper below selects its anomaly map.
    class ExportModel(torch.nn.Module):
        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image):
            result = self.wrapped(image)
            if isinstance(result, dict):
                value = result.get("anomaly_map", result.get("pred_score"))
            elif hasattr(result, "anomaly_map"):
                value = result.anomaly_map
            else:
                value = result
            if isinstance(value, (tuple, list)):
                value = value[0]
            return value

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 256, 256)
    torch.onnx.export(ExportModel(model), dummy, str(OUT), input_names=["image"],
                      output_names=["anomaly_map"], opset_version=17,
                      dynamic_axes={"image": {0: "batch"}, "anomaly_map": {0: "batch"}})
    print(f"Exported trained PatchCore graph to {OUT}")


if __name__ == "__main__":
    main()
