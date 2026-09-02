import pytest
import torch

pytest.importorskip("ultralytics")

from esmoe import attach_aux_loss, blocks, equip, graft, inject_esmoe  # noqa: E402
from esmoe.__main__ import main as cli  # noqa: E402
from esmoe.inject import AUX_NAME  # noqa: E402

BACKBONES = ["yolov8n.yaml", "yolo11n.yaml", "yolo12n.yaml"]
# Supported, but not shipped by every ultralytics release, so the tests skip what is absent.
GUARDED_BACKBONES = ["yolo26n.yaml", "yolov5n.yaml", "yolov9t.yaml", "yolov10n.yaml"]


def _model(base, nc=2, **graft_kwargs):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    inject_esmoe()
    cfg = graft(base, **graft_kwargs)
    cfg["nc"] = nc
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
    model.args = get_cfg(DEFAULT_CFG)  # loss() reads hyperparameters the trainer normally injects
    return model


def _count(items):
    # loss items are a named dict on ultralytics >= 8.4.13x and a tensor before that
    return len(items) if isinstance(items, dict) else items.numel()


def _batch():
    return {
        "img": torch.rand(2, 3, 64, 64),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }


def test_graft_appends_block_and_renumbers_head():
    from ultralytics.nn.tasks import yaml_model_load

    cfg = graft("yolov8n.yaml")
    plain = yaml_model_load("yolov8n.yaml")
    at = len(plain["backbone"])
    assert cfg["backbone"][-1][2] == "ESMoE"
    for before, after in zip(plain["head"], cfg["head"], strict=True):
        want = before[0]
        want = [w + 1 if isinstance(w, int) and w >= at else w for w in want] if isinstance(want, list) else want
        assert after[0] == want


def test_graft_accepts_several_insertion_points():
    from ultralytics.nn.tasks import yaml_model_load

    plain = yaml_model_load("yolov8n.yaml")
    cfg = graft("yolov8n.yaml", at=[4, 6])
    inserted = [layer for layer in cfg["backbone"] + cfg["head"] if layer[2] == "ESMoE"]
    assert len(inserted) == 2
    assert len(cfg["backbone"]) + len(cfg["head"]) == len(plain["backbone"]) + len(plain["head"]) + 2


def test_graft_rejects_impossible_insertion_points():
    with pytest.raises(ValueError):
        graft("yolov8n.yaml", at=999)
    with pytest.raises(ValueError):
        graft("yolov8n.yaml", at="middle")


@pytest.mark.parametrize("base", GUARDED_BACKBONES)
def test_newer_backbones_build_when_the_installed_release_ships_them(base):
    from ultralytics.utils import ASSETS  # noqa: F401  - import guard for a usable install

    try:
        model = _model(base)
    except FileNotFoundError:
        pytest.skip(f"{base} is not shipped by this ultralytics release")
    assert model(torch.rand(1, 3, 64, 64)) is not None


@pytest.mark.parametrize("base", BACKBONES)
def test_three_generations_build_and_forward(base):
    model = _model(base)
    assert next(blocks(model), None) is not None
    assert model(torch.rand(1, 3, 64, 64)) is not None


def test_multiple_blocks_build_and_forward():
    model = _model("yolov8n.yaml", at=[4, 6, 9])
    assert len(list(blocks(model))) == 3
    assert model(torch.rand(1, 3, 64, 64)) is not None


def test_attach_aux_loss_reaches_total_loss():
    model = _model("yolov8n.yaml")
    model.train()
    batch = _batch()

    torch.manual_seed(0)
    plain, plain_items = model.loss(batch)
    assert _count(plain_items) == 3

    attach_aux_loss(model, weight=1.0)
    torch.manual_seed(0)
    total, items = model.loss(batch)
    assert _count(items) == 4
    aux = items[AUX_NAME].item() if isinstance(items, dict) else items[-1].item()
    assert aux > 0
    assert total.sum().item() == pytest.approx(plain.sum().item() + aux * batch["img"].shape[0], rel=1e-4)
    total.sum().backward()
    block = next(blocks(model))
    assert sum(p.grad.abs().sum().item() for p in block.router.parameters() if p.grad is not None) > 0


def test_attach_refuses_a_model_without_blocks():
    from ultralytics.nn.tasks import DetectionModel

    with pytest.raises(ValueError):
        attach_aux_loss(DetectionModel("yolov8n.yaml", ch=3, nc=2, verbose=False))


def test_arm_re_attaches_to_the_rebuilt_trainer_model():
    from esmoe.inject import arm_trainer

    class Trainer:
        loss_names = ("box_loss", "cls_loss", "dfl_loss")

    trainer = Trainer()
    trainer.model = _model("yolov8n.yaml")
    arm = arm_trainer(0.05)
    arm(trainer)
    arm(trainer)
    assert trainer.loss_names.count(AUX_NAME) == 1
    assert trainer.model._esmoe_aux_weight == 0.05


def test_arm_leaves_empty_loss_names_to_the_trainer():
    from esmoe.inject import arm_trainer

    class Trainer:
        loss_names = ()

    trainer = Trainer()
    trainer.model = _model("yolov8n.yaml")
    arm_trainer(0.01)(trainer)
    # Releases that name losses from the returned dict start empty; appending here would make the
    # progress header show one column instead of four.
    assert trainer.loss_names == ()


def test_cli_writes_a_grafted_config(tmp_path):
    from ultralytics.utils import YAML

    out = tmp_path / "yolo11n-esmoe.yaml"
    assert cli(["graft", "yolo11n.yaml", "-o", str(out), "-e", "2", "-k", "1", "--at", "4,6"]) == 0
    cfg = YAML.load(str(out))
    assert [layer for layer in cfg["backbone"] if layer[2] == "ESMoE"][0][3] == [2, 1]
    assert cli(["info"]) == 0


def test_equip_builds_a_wired_model(tmp_path):
    out = tmp_path / "equipped.yaml"
    core = equip("yolov8n.yaml", weight=0.02, out=str(out)).model
    assert next(blocks(core), None) is not None
    assert core._esmoe_aux_weight == 0.02
    assert out.exists()


def test_equip_without_an_output_path_still_builds():
    core = equip("yolov8n.yaml", at=[4, 6]).model
    assert len(list(blocks(core))) == 2


def test_train_routes_to_a_trainer_that_lives_in_esmoe():
    from ultralytics import YOLO

    model = attach_aux_loss(YOLO(str(_grafted_yaml())), weight=0.02)
    cls = model._smart_load("trainer")
    assert cls.__module__ == "esmoe.trainer" and cls.__name__.startswith("ESMoE"), cls
    assert cls is model._smart_load("trainer"), "wrapping twice must hand back the same class"
    assert model._smart_load("validator").__module__.startswith("ultralytics"), "only the trainer is rerouted"


def test_a_fresh_interpreter_can_import_the_worker_trainer():
    """A DDP worker runs `from <module> import <Trainer>` in a fresh process; that line alone has to
    register the block and re-arm the auxiliary loss."""
    import os
    import subprocess
    import sys

    from ultralytics import YOLO

    model = attach_aux_loss(YOLO(str(_grafted_yaml())), weight=0.02)
    cls = model._smart_load("trainer")
    probe = (
        f"from {cls.__module__} import {cls.__name__} as T\n"
        "import ultralytics.nn.tasks as tasks, esmoe.inject as inject\n"
        "assert tasks.ESMoE is inject.ESMoE, 'block not registered in the worker'\n"
        "assert tasks.BaseModel.loss is inject._loss_with_aux, 'loss not patched in the worker'\n"
        "assert inject.weight() == 0.02, inject.weight()\n"
        "print(T.__mro__[1].__name__)\n"
    )
    done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, env=os.environ.copy())
    assert done.returncode == 0, done.stderr[-800:]
    assert done.stdout.strip().endswith("Trainer")


def _grafted_yaml():
    import tempfile
    from pathlib import Path

    inject_esmoe()
    target = Path(tempfile.mkdtemp(prefix="esmoe-test-")) / "v8-esmoe.yaml"
    graft("yolov8n.yaml", out=str(target))
    return target


def test_default_graft_leaves_the_p5_lateral_on_the_old_backbone_end():
    layers = (g := graft("yolov8n.yaml"))["backbone"] + g["head"]
    at = next(i for i, layer in enumerate(layers) if layer[2] == "ESMoE")
    lateral = next(layer for layer in layers[at + 1 :] if layer[2] == "Concat" and at - 1 in layer[0])
    assert lateral[0] == [-1, at - 1], "the P5 lateral still reads SPPF, so the block only feeds the top-down path"


def test_rewire_points_every_consumer_of_the_insertion_layer_at_the_block():
    for base in BACKBONES:
        layers = (g := graft(base, rewire=True))["backbone"] + g["head"]
        at = next(i for i, layer in enumerate(layers) if layer[2] == "ESMoE")
        later = layers[at + 1 :]
        stale = [layer for layer in later if (at - 1 in layer[0] if isinstance(layer[0], list) else layer[0] == at - 1)]
        assert not stale, f"{base}: {stale} still read the layer before the block"
        by_index = any(isinstance(layer[0], list) and at in layer[0] for layer in later)
        assert by_index, f"{base}: nobody reads the block by index"


@pytest.mark.parametrize("base", BACKBONES + GUARDED_BACKBONES)
def test_rewired_models_build_and_forward(base):
    try:
        model = _model(base, rewire=True)
    except FileNotFoundError:
        pytest.skip(f"{base} is not shipped by this ultralytics release")
    out = model(torch.rand(1, 3, 64, 64))
    # Backbone-end width differs per generation (256 on v8/11/12/26, 128 on v9t); built is what matters.
    assert out is not None and next(blocks(model)).channels
