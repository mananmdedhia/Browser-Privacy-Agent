"""Export the trained tagger to ONNX so it can run inside the browser (onnxruntime-web), with no server at all.
Then check the ONNX model gives the same numbers as the PyTorch one.

Run:  python -m privylens.engine.export_onnx            -> checkpoints/tagger.onnx
Input  ids    (batch, tokens) int64
Output logits (batch, tokens, 23)
"""
import sys
import numpy as np
import onnxruntime as ort
import torch
from privylens.models.encoder_tagger import load_tagger


def export(ckpt="checkpoints/tagger.pt", out="checkpoints/tagger.onnx"):
    model = load_tagger(ckpt)
    ids = torch.randint(6, 1000, (1, 20))
    torch.onnx.export(model, (ids,), out, input_names=["ids"], output_names=["logits"],
                      dynamic_axes={"ids": {0: "batch", 1: "tokens"}, "logits": {0: "batch", 1: "tokens"}},
                      opset_version=17, dynamo=False)
    # same input, both runtimes: results must match
    test = torch.randint(6, 1000, (2, 33))                      # a different length than the export example
    with torch.no_grad():
        want = model(test).numpy()
    got = ort.InferenceSession(out).run(None, {"ids": test.numpy()})[0]
    diff = float(np.abs(want - got).max())
    assert diff < 1e-4, f"ONNX differs from PyTorch by {diff}"
    print(f"saved {out}: matches PyTorch (max difference {diff:.2e})")


if __name__ == "__main__":
    export(*sys.argv[1:])
