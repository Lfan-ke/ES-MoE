import numpy as np
import pytest
import torch

import esmoe

ort = pytest.importorskip("onnxruntime")


def _routing_by_sign(channels=16):
    """A block whose routed expert follows the sign of the input, so a trace taken on one input is
    only correct for the other if the block did not bake its routing into the graph."""
    block = esmoe.ESMoE(num_experts=4, top_k=1, channels=channels).eval()
    with torch.no_grad():
        block.router[2].weight.zero_()
        block.router[2].bias.zero_()
        block.router[2].weight[0].fill_(4.0)
        block.router[2].weight[1].fill_(-4.0)
        block.router[4].weight.zero_()
        block.router[4].bias.zero_()
        block.router[4].weight[0, 0] = 6.0
        block.router[4].weight[1, 1] = 6.0
    return block


def test_export_stays_faithful_for_inputs_that_route_elsewhere(tmp_path):
    block = _routing_by_sign()
    positive = torch.full((1, 16, 4, 4), 0.7)
    negative = torch.full((1, 16, 4, 4), -0.7)
    assert torch.softmax(block.router(positive), 1).argmax() != torch.softmax(block.router(negative), 1).argmax()

    path = tmp_path / "block.onnx"
    # The TorchScript exporter is the one that traces, and the one ultralytics still uses; it is
    # also the path where a data-dependent skip would be baked into the graph.
    torch.onnx.export(
        block, (positive,), str(path), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])

    for sample in (positive, negative):
        with torch.no_grad():
            expected = block(sample).numpy()
        exported = session.run(["y"], {"x": sample.numpy()})[0]
        assert np.abs(expected - exported).max() < 1e-4
