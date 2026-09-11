"""The ``"upstream"`` recipe against YOLO-Master's trainer, its operations rewritten here from source.

`nn/mixture_loss.py` (`_update_mixture_loss_ema_batch`, `_collect_mixture_aux_loss`) composes the
routed term, `engine/trainer.py` (`build_optimizer`) groups the routers and
`engine/extensions/mixture.py` (`begin_epoch`) holds the experts back. Upstream composes four routed
families where this package has one and keeps its running mean in a float32 buffer rather than a
float; the numbers still have to agree.
"""

import csv
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("ultralytics")

import esmoe  # noqa: E402
from esmoe import inject, upstream  # noqa: E402
from esmoe.inject import AUX_NAME  # noqa: E402

DEFAULTS = (1.0, 0.1, 0.1, 0.1)  # running-mean starts for moe, mot, moa, latent
GAINS = (1.0, 1.0, 1.0, 0.1)


@pytest.fixture(autouse=True)
def _process_settings(monkeypatch):
    # attach_aux_loss arms the whole process and its environment; both go back for the next test.
    monkeypatch.setattr(inject, "_WEIGHT", inject._WEIGHT)
    monkeypatch.setattr(inject, "_RECIPE", inject._RECIPE)
    for key in (inject.ENV_WEIGHT, inject.ENV_RECIPE):
        monkeypatch.setenv(key, os.environ.get(key, ""))
        if not os.environ[key]:
            monkeypatch.delenv(key)


def fork_compose(buffer: torch.Tensor, moe: torch.Tensor, gain: float = 1.0, budget: float = 3.0):
    """Upstream's isolation, running-mean update and composition, the other routed families at zero."""
    moe = moe if torch.isfinite(moe) else moe.new_zeros(())
    losses = (moe, torch.zeros(()), torch.zeros(()), torch.zeros(()))
    values = torch.stack([loss.detach().abs().reshape(()) for loss in losses])
    with torch.no_grad():
        safe = torch.where(torch.isfinite(buffer), buffer, torch.tensor(DEFAULTS))
        buffer = safe.mul(0.99).add(values.clamp(1e-4, 1e4) * 0.01)
    scales = [min(max(float(value), 1e-6), 1e6) for value in buffer.tolist()]
    terms = [loss / scale * g for loss, scale, g in zip(losses, scales, (gain, *GAINS[1:]), strict=True)]
    observed = torch.stack([term.detach().abs() for term in terms])
    factor = torch.minimum(torch.tensor(1.0), torch.tensor(budget) / observed.sum().clamp_min(1e-4)).detach()
    result = sum(terms) * factor
    if not torch.isfinite(result).all() or torch.abs(result) > 1e4:
        return moe.new_zeros(()), buffer
    return result, buffer


@pytest.mark.parametrize("weight", [1.0, 0.25])
def test_the_routed_term_is_composed_step_by_step_as_upstream_composes_it(weight):
    owner, buffer = nn.Module(), torch.tensor(DEFAULTS)
    # Collapsed routing under the cap, a calm stretch, a zero, a spike past the clamp, a non-finite step.
    for value in (4.0, 3.9, 3.6, 2.2, 1.4, 0.0, 5e4, float("nan"), 1.1, 1.05):
        theirs = torch.tensor(value, requires_grad=True)
        ours = torch.tensor(value, requires_grad=True)
        expected, buffer = fork_compose(buffer, theirs, gain=weight)
        got = upstream.normalise(owner, ours, weight)
        assert got.item() == pytest.approx(expected.item(), rel=1e-5, abs=1e-8)
        assert owner._esmoe_aux_scale == pytest.approx(float(buffer[0]), rel=1e-5)
        if expected.requires_grad:
            (want,) = torch.autograd.grad(expected, theirs)
            (have,) = torch.autograd.grad(got, ours)
            assert have.item() == pytest.approx(want.item(), rel=1e-5)


def test_the_mean_starts_at_one_and_moves_before_it_divides():
    owner = nn.Module()
    term = upstream.normalise(owner, torch.tensor(2.0), 1.0)
    assert owner._esmoe_aux_scale == pytest.approx(1.01)
    assert term.item() == pytest.approx(2.0 / 1.01)


def _model(**graft_kwargs):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    esmoe.inject_esmoe()
    cfg = esmoe.graft("yolov8n.yaml", **graft_kwargs)
    cfg["nc"] = 2
    model = DetectionModel(cfg, ch=3, nc=2, verbose=False)
    model.args = get_cfg(DEFAULT_CFG)
    return model


def _batch():
    return {
        "img": torch.rand(2, 3, 64, 64),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }


def _aux(items) -> float:
    return (items[AUX_NAME] if isinstance(items, dict) else items[-1]).item()


def test_the_term_reaches_every_native_term_once_and_never_per_image():
    """Upstream adds a scalar to the (box, cls, dfl) vector the criterion returns already scaled by the
    batch, and the trainer sums the vector: the term counts once per native term, not once per image."""
    model = _model()
    model.train()
    batch = _batch()
    plain, _ = model.loss(batch)
    esmoe.attach_aux_loss(model, weight=1.0, recipe="upstream")
    total, items = model.loss(batch)
    aux = _aux(items)
    assert 0 < aux <= upstream.BUDGET
    assert total.shape == plain.shape
    assert total.sum().item() == pytest.approx(plain.sum().item() + plain.numel() * aux, rel=1e-4)


def test_outside_training_the_term_is_left_out_and_the_mean_stays_put():
    model = _model()
    esmoe.attach_aux_loss(model, weight=1.0, recipe="upstream")
    model.eval()
    _, items = model.loss(_batch())
    assert _aux(items) == 0.0
    assert not hasattr(model, "_esmoe_aux_scale")


@pytest.mark.parametrize("name,iterations", [("auto", 2e4), ("auto", 1e3), ("SGD", 1e5)])
def test_routers_train_at_half_the_rate_and_outside_muon(name, iterations):
    from ultralytics.models.yolo.detect import DetectionTrainer

    model = _model()
    trainer = SimpleNamespace(args=SimpleNamespace(lr0=0.01, momentum=0.937, warmup_bias_lr=0.1), data={"nc": 2})
    optimizer = DetectionTrainer.build_optimizer(trainer, model, name, 0.01, 0.937, 5e-4, iterations)
    count = sum(len(group["params"]) for group in optimizer.param_groups)
    base = min(group["lr"] for group in optimizer.param_groups)

    moved = upstream.split_routers(optimizer, model, 5e-4)

    routers = {id(p) for block in esmoe.blocks(model) for p in block.router.parameters()}
    holding = [group for group in optimizer.param_groups if any(id(p) in routers for p in group["params"])]
    assert len(holding) == 1 and moved == len(routers) == len(holding[0]["params"])
    group = holding[0]
    assert group["lr"] == pytest.approx(upstream.ROUTER_LR * base)
    assert group["weight_decay"] == 5e-4 and not group.get("use_muon")
    assert sum(len(group["params"]) for group in optimizer.param_groups) == count


def test_experts_wait_three_epochs_and_nothing_else_does():
    model = _model(out_norm=True)
    block = next(esmoe.blocks(model))
    frozen = next(block.experts.parameters())
    frozen.requires_grad = False  # a layer the run froze on its own
    trainer, trained = SimpleNamespace(model=model, epoch=0), []
    for epoch in range(5):
        trainer.epoch = epoch
        inject.warm_experts(trainer)
        trained.append(all(p.requires_grad for p in block.experts.parameters() if p is not frozen))
        assert all(p.requires_grad for p in (*block.router.parameters(), *block.norm.parameters()))
        assert not frozen.requires_grad
    assert trained == [False, False, False, True, True]


def test_a_trainer_that_never_split_the_routers_is_refused():
    trainer = SimpleNamespace(model=_model(), loss_names=())
    with pytest.raises(RuntimeError, match="router"):
        inject.arm_trainer(1.0, "upstream")(trainer)


def test_an_unknown_recipe_is_refused():
    with pytest.raises(ValueError, match="recipe"):
        esmoe.attach_aux_loss(_model(), recipe="paper")


def test_a_worker_process_inherits_the_recipe(tmp_path):
    from ultralytics import YOLO

    esmoe.inject_esmoe()
    cfg = tmp_path / "worker.yaml"
    esmoe.graft("yolov8n.yaml", out=str(cfg))
    model = esmoe.attach_aux_loss(YOLO(str(cfg)), weight=1.0, recipe="upstream")
    cls = model._smart_load("trainer")
    probe = (
        f"from {cls.__module__} import {cls.__name__} as T\n"
        "import esmoe.inject as inject\n"
        "assert inject.recipe() == 'upstream', inject.recipe()\n"
        "assert T.build_optimizer.__module__ == 'esmoe.trainer'\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=os.environ.copy())
    assert done.returncode == 0, done.stderr[-800:]


def test_a_trained_model_gets_the_whole_recipe(tmp_path):
    from ultralytics import YOLO

    esmoe.inject_esmoe()
    cfg = tmp_path / "upstream.yaml"
    esmoe.graft("yolov8n.yaml", out=str(cfg), balance="gshard", out_norm=True, dense_training=True)
    model = esmoe.attach_aux_loss(YOLO(str(cfg)), weight=1.0, recipe="upstream")
    seen: dict = {"frozen": set()}

    def on_train_start(trainer):
        routers = {id(p) for block in esmoe.blocks(trainer.model) for p in block.router.parameters()}
        groups = trainer.optimizer.param_groups
        holding = [group for group in groups if any(id(p) in routers for p in group["params"])]
        seen["routers"] = [(group.get("param_group"), group["lr"]) for group in holding]
        seen["base"] = min(group["lr"] for group in groups if group.get("param_group") != "router")

    def on_train_batch_end(trainer):
        experts = [p for block in esmoe.blocks(trainer.model) for p in block.experts.parameters()]
        seen["frozen"].add((trainer.epoch, not any(p.requires_grad for p in experts)))

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.train(
        data="coco8.yaml",
        epochs=1,
        imgsz=64,
        batch=2,
        workers=0,
        device="cpu",
        pretrained=False,
        plots=False,
        val=False,
        project=str(tmp_path),
        name="run",
        exist_ok=True,
        verbose=False,
    )

    ((label, lr),) = seen["routers"]
    assert label == "router" and lr == pytest.approx(upstream.ROUTER_LR * seen["base"])
    assert seen["frozen"] == {(0, True)}
    with open(Path(model.trainer.save_dir) / "results.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    key = next(k for k in rows[0] if k.strip().endswith(AUX_NAME))
    assert all(0 < float(row[key]) <= upstream.BUDGET for row in rows)
