# ES-MoE Toolkit - E1 Roadmap

腾讯犀牛鸟 YOLO-Master 项目阶段课题 E1(社区影响力 / 开源工程化)。把 ES-MoE 的"民间涨点"做成官方、可安装、预算公平、正确接入 aux loss 的能力。

## Baselines (locked)

- 验收 Release: YOLO-Master-v26.08 @ 43d4011
- 研究 HEAD: 57b9ea3
- ES-MoE-N 锚点: 2.68M params / 8.7 GFLOPs / 42.7 mAP

## Goals

- P0 (保底): 一个预算下 3-seed 复选 ES-MoE-N 并选型;外部 Ultralytics 主干最小接入;aux loss 非零日志。
- P1 (预期): pip / 配置注入式插件,兼容 YOLOv8 / YOLO11 / YOLOv12;README + Colab。
- P2 (理想): 发布 PyPI 独立包;中英教程;按社区反馈迭代。

## 8.24 准入 (smoke)

- 外部 YOLO 主干完成 ESMoE 最小接入。
- collect_aux_loss 输出非零日志。
- API 草案(inject_esmoe / ESMoE / collect_aux_loss)通过评审。

## Schedule (8.14-9.14)

- 8.22-8.23 锁 commit、读切入点、跑准入 smoke
- 8.24 环境检查、命令 / 日志 / 配置、P0 与跨级方案
- 8.25-8.31 最小闭环自复现数字,每日可回 checkpoint
- 9.1-9.7 关键涨点与中期演示;PR 基线
- 9.8-9.12 多 seed 补实验;冻结复现包
- 9.13-9.14 结项答辩;最终 PR + 报告 + 已知局限

## Sources

ES-MoE 模块与 collect_aux_loss 迁自 YOLO-Master 的 #52 工作(本机 fork tencent-yolo-master);实验脚本见 yolomoe-ln/scripts。
