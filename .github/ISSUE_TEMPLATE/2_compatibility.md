---
name: Compatibility break
about: A new ultralytics or torch release broke the plugin.
title: '[Compat] ultralytics X.Y.Z '
labels: compatibility
assignees: ''
---

<!-- This project only claims to work with stock ultralytics, so upstream releases are the usual
     cause of breakage. Reports here are welcome even without a fix. -->

### Versions

- **esmoe**:
- **ultralytics that works**:
- **ultralytics that breaks**:
- **torch**:

### What breaks

<!-- Which of the three integration points fails: model construction from a grafted config, the
     auxiliary loss reaching the trainer, or the CLI. Include the error. -->

### Minimal reproduction

```python
import esmoe
model = esmoe.equip("yolo11n.yaml")
```
