"""DDP without `find_unused_parameters`, which ultralytics picks whenever `compile=True`.

Sparse dispatch leaves an expert that no image in the batch routed to out of the autograd graph.
With the unused-parameter search off, DDP's reducer then fails on the next step. It takes two real
ranks to see it: a single rank never waits on a collective.
"""

import tempfile
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from esmoe import ESMoE

pytestmark = pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="needs gloo")


def _block() -> ESMoE:
    torch.manual_seed(0)
    block = ESMoE(4, 2, channels=8)
    with torch.no_grad():
        # Experts 2 and 3 are never routed to, on either rank, at any step.
        block.router[-1].bias.copy_(torch.tensor([5.0, 5.0, -30.0, -30.0]))
    return block


def _rank(rank: int, store: str, static_graph: bool, box) -> None:
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        block = _block()
        ddp = DistributedDataParallel(block, find_unused_parameters=False, static_graph=static_graph)
        torch.manual_seed(rank + 1)
        x = torch.randn(4, 8, 6, 6)
        for _ in range(3):
            ddp.zero_grad()
            ddp(x).sum().backward()
        idle = [p.grad for expert in block.experts[2:] for p in expert.parameters()]
        box[rank] = "ok" if all(g is not None and not g.any() for g in idle) else "an unrouted expert has no gradient"
    except Exception as exc:
        box[rank] = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("static_graph", [False, True])
def test_two_ranks_train_through_unrouted_experts_without_the_unused_parameter_search(static_graph):
    ctx = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp, ctx.Manager() as manager:
        box = manager.dict()
        store = (Path(tmp) / "store").as_uri()
        ranks = [ctx.Process(target=_rank, args=(r, store, static_graph, box)) for r in range(2)]
        for p in ranks:
            p.start()
        for p in ranks:
            p.join(180)
        for p in ranks:
            if p.is_alive():
                p.terminate()
        report = dict(box)
    # Zero is the gradient DDP writes for a parameter it found unused, so the update is unchanged.
    assert report == {0: "ok", 1: "ok"}


def test_the_block_compiles_and_agrees_with_eager():
    """`compile=True` hands the model to TorchDynamo before DDP wraps it; the compiled block has to
    train and produce what the eager one does."""
    pytest.importorskip("torch._dynamo")
    torch.manual_seed(0)
    block = ESMoE(4, 2, channels=16).train()
    compiled = torch.compile(block, backend="aot_eager")
    x = torch.randn(4, 16, 8, 8)
    compiled(x).sum().backward()
    eager = ESMoE(4, 2, channels=16).train()
    eager.load_state_dict(block.state_dict())
    assert torch.allclose(compiled(x), eager(x), atol=1e-5)


def test_outside_a_process_group_unrouted_experts_stay_out_of_the_graph():
    """A single process keeps skipping them outright, so every recorded run still reproduces: an
    optimiser leaves a parameter with no gradient alone, while a zero gradient would still decay."""
    block = _block()
    block(torch.randn(4, 8, 6, 6)).sum().backward()
    for expert in block.experts[2:]:
        assert all(p.grad is None for p in expert.parameters())
