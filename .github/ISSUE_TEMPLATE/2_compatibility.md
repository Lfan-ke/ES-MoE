---
name: Compatibility break
about: A new ultralytics or torch release broke the plugin.
title: '[Compat] ultralytics X.Y.Z '
labels: compatibility
assignees: ''
---

<!-- The plugin only claims stock ultralytics, so upstream releases are the usual cause. A report without a fix is welcome. -->

### Versions

- **esmoe**:
- **ultralytics that works**:
- **ultralytics that breaks**:
- **torch**:

### What breaks

- [ ] Building a model from a grafted config
- [ ] The auxiliary loss reaching the trainer
- [ ] The `esmoe` command line

<!-- Paste the error. -->

### Minimal reproduction

```python
import esmoe

model = esmoe.equip("yolo11n.yaml")
```
