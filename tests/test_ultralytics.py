import pytest
import torch

pytest.importorskip("ultralytics")

from esmoe import attach_aux_loss, blocks, equip, graft, inject_esmoe  # noqa: E402
from esmoe.__main__ import main as cli  # noqa: E402
from esmoe.inject import AUX_NAME  # noqa: E402

BACKBONES = ["yolov8n.yaml", "yolo11n.yaml", "yolo12n.yaml"]
# Shipped by newer ultralytics only, and not yet covered by a training run of its own.
FUTURE_BACKBONES = ["yolo26n.yaml"]


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


@pytest.mark.parametrize("base", FUTURE_BACKBONES)
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
    from esmoe.inject import _arm

    class Trainer:
        loss_names = ("box_loss", "cls_loss", "dfl_loss")

    trainer = Trainer()
    trainer.model = _model("yolov8n.yaml")
    arm = _arm(0.05)
    arm(trainer)
    arm(trainer)
    assert trainer.loss_names.count(AUX_NAME) == 1
    assert trainer.model._esmoe_aux_weight == 0.05


def test_arm_leaves_empty_loss_names_to_the_trainer():
    from esmoe.inject import _arm

    class Trainer:
        loss_names = ()

    trainer = Trainer()
    trainer.model = _model("yolov8n.yaml")
    _arm(0.01)(trainer)
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
