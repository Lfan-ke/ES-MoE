def inject_esmoe():
    # Expose ESMoE where ultralytics parse_model resolves layer names, so model.yaml can
    # reference `ESMoE` across YOLOv8 / YOLO11 / YOLOv12 backbones.
    from .module import ESMoE
    import ultralytics.nn.tasks as tasks
    setattr(tasks, "ESMoE", ESMoE)
    return ESMoE
