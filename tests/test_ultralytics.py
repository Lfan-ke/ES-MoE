import pytest
import torch

pytest.importorskip("ultralytics")

from esmoe import ESMoE, attach_aux_loss, graft, inject_esmoe  # noqa: E402
from esmoe.inject import AUX_NAME  # noqa: E402

BACKBONES = ["yolov8n.yaml", "yolo11n.yaml", "yolo12n.yaml"]


def _model(base, nc=2):
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import DEFAULT_CFG

    inject_esmoe()
    cfg = graft(base)
    cfg["nc"] = nc
    model = DetectionModel(cfg, ch=3, nc=nc, verbose=False)
    model.args = get_cfg(DEFAULT_CFG)  # loss() reads hyperparameters the trainer normally injects
    return model


def _batch(nc=2, imgsz=64):
    return {
        "img": torch.rand(2, 3, imgsz, imgsz),
        "cls": torch.zeros(2, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.1]]),
        "batch_idx": torch.tensor([0.0, 1.0]),
    }


def test_graft_appends_block_and_renumbers_head():
    cfg = graft("yolov8n.yaml")
    assert cfg["backbone"][-1][2] == "ESMoE"
    from ultralytics.nn.tasks import yaml_model_load

    plain = yaml_model_load("yolov8n.yaml")
    at = len(plain["backbone"])
    for before, after in zip(plain["head"], cfg["head"]):
        want = before[0]
        want = [w + 1 if isinstance(w, int) and w >= at else w for w in want] if isinstance(want, list) else want
        assert after[0] == want


@pytest.mark.parametrize("base", BACKBONES)
def test_three_generations_build_and_forward(base):
    model = _model(base)
    assert any(isinstance(m, ESMoE) for m in model.modules())
    out = model(torch.rand(1, 3, 64, 64))
    assert out is not None


def test_attach_aux_loss_reaches_total_loss():
    model = _model("yolov8n.yaml")
    model.train()
    batch = _batch()

    torch.manual_seed(0)
    plain, plain_items = model.loss(batch)
    assert plain_items.numel() == 3

    attach_aux_loss(model, weight=1.0)
    torch.manual_seed(0)
    total, items = model.loss(batch)
    assert items.numel() == 4
    aux = items[-1].item()
    assert aux > 0
    batch_size = batch["img"].shape[0]
    assert total.sum().item() == pytest.approx(plain.sum().item() + aux * batch_size, rel=1e-4)
    total.sum().backward()
    block = next(m for m in model.modules() if isinstance(m, ESMoE))
    assert sum(p.grad.abs().sum().item() for p in block.router.parameters() if p.grad is not None) > 0


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
