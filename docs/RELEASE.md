# 发布说明

当前版本 **1.0.1**：文档站为每一项实验结果配图，按上游代码与论文逐条核对对照，并补齐包的元数据与发布检查。公开接口与 1.0.0 相同。

## 修复

- **交付审计只认确切的声明。** `scripts/closure.py` 判断不满三个 seed 的格子是否声明过，原先按子串匹配，一个更长的臂名就能让较短的臂名算作已声明；现在要求实验页中英两版都写出带反引号的完整臂名。
- **数据集统计更稳。** `scripts/dataset.py` 遇到略微越过图像边界的标注框不再中断；类名写成列表或映射都能读；压缩包读完即关闭；`.txt` 与 `.zip` 后缀不分大小写；新增 `--out`。
- **图表只取登记过的运行。** `scripts/charts.py` 画同配置对照的路由图时，只读各 seed 登记的 B 臂，同一 seed 的重复运行不会多出一行；读取的结果表缺失或解析为空时直接报错。
- **文档站的几处显示。** 站点根的 404 页显示英文；`latest/` 下的旧地址跳到带版本号的副本；暗色模式下流程图仍是亮色；手机上图表拥挤；非最新版本的页面没有提示。

## 文档

- **每项结果都有图。** 实验页的 25 张交互图覆盖数据集、训练协议、选型、七代主干、面积分档、对照臂、同配置对照、逐步对拍、发布模型复核、重复运行、路由与平衡目标、训练开销；选型、实验结果与效果图页嵌入对应的图。图表跟随暗色模式，窄屏自动收紧。
- **对照上游与论文。** 「ES-MoE 与 YOLO」页按 YOLO-Master `acce839c` 的源码与论文逐条核对后重写，更正块分析次数、发布模型复核的精度表述、验证时平衡项是否计入等十余处，补上多卡、精度回退、导出前剪枝与论文损失设置的对照。
- **API 与教程。** API 页补上四个平衡目标、`DWExpert`、`blocks()` 与命令行；教程的自定义训练循环补上 `clear_aux_loss()`；README 的参数与卡时开销按主干给出区间。
- **图表数据由脚本现算。** `scripts/dataset.py` 从数据集压缩包统计出 `results/dataset.json`；`scripts/charts.py` 写出各图所用的数据。
- **按 1.0.1 重跑的产物。** `results/verify.json` 十项检查全部通过；快速上手笔记本的输出来自在 Linux 上安装 PyPI 1.0.1 后的一次运行。

## 仓库

- **包的元数据。** 许可证写成 SPDX 表达式 `AGPL-3.0-only`，PyPI 页面据此显示许可证；补上 Python 3.10 至 3.12 的分类。
- **发布与 CI 检查。** 标签与 `esmoe.__version__` 不一致时发布失败；CI 严格构建文档站；手动重部署文档只接受 main 上已打标签的版本，且只有最新标签会移动 `latest`。
- **测试。** 新增图表数据、数据集统计、审计声明与版本号一致性的测试，共 335 项。
- **检查点。** `checkpoints` 分支补上 43 次早先协议运行的 `last.pt`，共 72 个；每个都先核对同一运行的 `best.pt` 与记录和分支上的哈希一致。
- **旧版本文档。** 0.1.6 的文档保留，每页顶部提示它不是最新发布版。

## 此前的 1.0.0

首个稳定版。`esmoe.__all__` 里的公开接口此后按语义化版本维护：不兼容的改动只随主版本号出现。

### 新增

- **中断的运行可以接着跑完。** `scripts/train.py --resume <last.pt>` 从运行自己的检查点接着训练，不从头再来；只接受本运行目录里、由本运行保存的检查点。记录新增 `resumed` 字段，写检查点路径、哈希、已完成的 epoch 数与这些 epoch 的用时；`budget.gpu_hours` 是两段之和，`budget.epochs_replayed` 按余下的 epoch 计。
- **文档站按版本切换。** 每个发布版有自己的一份文档，`latest` 指向最新发布版，`dev` 跟随 main；原来不带版本的地址都跳到 `latest`。

### 修复

- **`scripts/measure.py` 重建不了续跑过的运行。** 续跑会把运行参数里的 `model` 改写成检查点路径，重测按它重建模型时报错。现在改用检查点记下的配置。
- **失败的运行在队列日志里记为完成。** `scripts/train.py` 写完失败记录后以非零码退出，`scripts/queue.sh` 据此记为 FAILED。
- **同一配置再跑一遍，会顶替原来的实验。** `scripts/report.py` 与 `scripts/same_config.py` 在同一配置、同一 seed 的记录里保留最早开跑的那条作为实验，后跑的只进噪声底；噪声底按硬件栈与预算分组，预算含精度，FP32 与混合精度分开计算。
- **文档站的几处入口。** 中文效果图页显示英文图；实验结果页的目录被嵌入的表格打断；`latest` 首页切换语言跳到旧版本；404 页取不到搜索与版本列表；各版本 sitemap 的语言链接少一个斜杠。
- **交付审计与平衡压力表。** `scripts/closure.py` 按章节统计判读线页上的登记，正文里提到「预登记」不再重复计数；`scripts/pressure.py` 按主干分组时把 `yolo-master-n` 当作一个名字，表按现有全部路由记录重算。
- **快速上手的「对齐上游」少了两项。** 补上 `balance="gshard"` 与 `recipe="upstream", weight=1.0`；缺 pandas 时一并安装。

### 数据

- 与 YOLO-Master 的同配置对照两轮入库：第七轮按各框架默认精度，第八轮四臂全程 FP32，另有 FP32 同卡重复运行量出的噪声底。判定见[判读线](JUDGMENT.md)。

### 仓库

- 根目录只留包与 GitHub 需要的文件：`uv.lock` 不再入库，CI 按当天解析到的依赖测试；文档站配置移到 `.github/docs/`；贡献指南与行为准则移到 `.github/`；环境快照移到 `results/env/`。
- 选型页补中文版（英文在 `SELECTION.en.md`）；issue 与 PR 模板加上必填说明与确认清单；README 的图片与许可证改用绝对链接，PyPI 页面上也能显示；README 与教程里的 `scripts/sweep.sh` 改为 `uv run bash` 调用，脚本里的 `python3` 取到的是项目环境；CITATION 的作者写成 `Cheng, Leo`，GitHub 生成的 APA 与 BibTeX 引用都是这一写法。

## 此前的 0.1.6

### 修复

- **`dynamic_threshold` 剪错了专家。** 阈值原先与 top-k 重归一**之前**的原始概率比较，上游比的是重归一**之后**的份额（`_soft_top_k` 先归一，`_sparse_forward` 再比）。四选二时四个概率加起来是一，第二名很少能到 0.4，于是几乎每张图都被剪成单专家，而训练走的是两专家混合。实测：一个 VisDrone 上训练好的四块模型，同一份权重，剪错时 mAP50 0.0427，按上游次序剪 0.3741，不剪 0.3748。默认 `dynamic_threshold=0.0` 不剪，`results/` 里的记录不受影响。

### 新增

- **`recipe="upstream"`。** `attach_aux_loss` 与 `equip` 的新参数，按 YOLO-Master 训练器对含路由模块模型的做法训练，供与上游同配置对比。三件事：
  - 辅助项除以自身幅值的滑动平均（衰减 0.99，初值 1.0），乘 `weight` 后封顶 3.0，加到 box、cls、dfl 三项上各一次；
  - 路由器参数单独成组，学习率减半、不进 Muon；
  - 前 3 个 epoch 冻结专家参数。

  常数与出处在 `esmoe.upstream`，逐步数值对上游源码的测试在 `tests/test_recipe.py`。默认 `"esmoe"` 不变，已有记录照旧可复现。
- **在 YOLO-Master 的分支上跑同一协议。**
  - `scripts/train.py` 新增：`--upstream` 训练上游自己的块，`--grafted` 训练已含 `ESMoE` 的配置，`--recipe` 选择训练方式。
  - 记录新增字段：`git_ref.framework`，以及 `budget.amp_at_end`、`budget.batch_at_end`、`budget.epochs_replayed`。上游训练器会在首个非有限梯度后关闭混合精度并重跑该 epoch，两个训练器都会在首个 epoch 显存不足时把 batch 减半，而训练参数不反映这些。
  - `scripts/report.py` 的分组键加入框架。
  - `scripts/same_config.py` 出同配置对比表。
  - `configs/yolo-master-n.yaml` 是上游模型去掉四个块的基线，`configs/yolo-master-n-esmoe.yaml` 与上游模型逐层参数一致。

## 此前的 0.1.5

### 新增

- **与上游 `ES_MOE` 的参数对齐。** 块现在接上游构造函数的全部参数：`out_channels`、`top_k=None`（等于用全部专家）、`sparse_inference`（上游的 `use_sparse_inference`）、`dynamic_threshold`（上游 0.4，本包默认 0.0 不剪——`results/` 里每条记录都是这样量出来的）。偶数核逐一降为奇数再按 `max_kernel_size` 截断，剪枝过的检查点因此装得回去；`num_experts` / `reduction` / `dynamic_threshold` / `max_kernel_size` 在构造时就按同样的边界校验。
- **四个平衡目标，默认仍是 Switch。** `switch_balance`（默认，与 0.1.4 和 `results/` 里的全部记录一致）、`gshard_balance`（对齐上游：读 top-k 掩码重归一后的门控）、`master_balance`（论文式 13，与 `gshard` 只差仿射 `(L−1)/E²`）、`gshard_probs_balance`（读原始概率，用来隔离「读哪个张量」这一个变量）。默认值由数据定：读门控的目标对没进 top-k 的专家梯度恒为零，实测 6 个检查点里 5 个出现死专家，Switch 在 66 个里一个没有（判读线第六轮）。命令行 `--balance {switch,gshard,master,gshard_probs}`。
- **多卡与 `compile=True` 同用。** 进程组里没被路由到的专家以零权重留在计算图中，DDP 不再依赖 `find_unused_parameters`。ultralytics 在 `compile=True` 时用的 `find_unused_parameters=False`、`static_graph=True` 已用两个 gloo 进程验过，块在 TorchDynamo 下能编译且与 eager 一致（`tests/test_distributed.py`）。单进程行为不变。
- **自定义平衡目标与专家写进配置。** `graft(balance=fn, expert=cls)` 把自定义的函数与类写成 `模块:限定名`，训练器与每个 DDP 子进程重建模型时都从这个名字导入回同一个对象；按名导入不回来的（lambda、嵌套函数、`__main__` 里定义的）在嫁接时就拒绝。内置专家有了短名 `dw`，`esmoe.EXPERTS` 与 `esmoe.BALANCES` 并列。
- **改宽的块嫁接进官方 yaml。** `graft(out_channels=N)` 在块后写一个官方 `Index` 层，块把输出包成单元素列表交给它；官方 `parse_model` 读这一层声明的宽度，下游因此按真实宽度建。`N` 是字面值，不随 width 倍率缩放。
- **命令行补齐。** `esmoe graft` 新增 `--out-channels`、`--balance`、`--expert`、`--out-norm`、`--dense-training`。
- **上游的块布局与块内结构。** `graft(at="backbone_stages")` 在主干每个 stage 后各放一块，七代主干均得四块（stage 边界由下采样层反推）；`out_norm` 补上加权求和后的 `BatchNorm + SiLU`（论文式 2 的 `Norm`）；`dense_training` 让训练期跑满专家，未选的权重为 0 但归一化统计量继续更新。三者默认关着，以保 `results/` 里既有的运行可原样复现。
- **`scripts/blockspec.py`**：从任一检查点读回当时真正生效的块设置。**`scripts/backfill.py`**：补齐并复核记录里的配置 hash、数据集样本数、GPU-hours、产物校验和，`--settings` 按权重改写记录并把改动写进记录本身。**`scripts/queue.sh`** 入库，与 `scripts/train.py` 的运行命名由测试对表。
- `scripts/report.py` 的配对差值附 95% 置信区间；`scripts/routing.py` 逐块分析而非只看第一块。

### 修复

- **块设置进不了训练的那个模型。** 训练器照 `model.yaml` 重建模型，`YOLO(cfg)` 之后设到块上的目标函数与开关随那个被丢弃的实例一起消失，不报错也不留痕——`--balance`、`--out-norm`、`--dense-training` 因此全部失效，而记录照命令行写。现在 `graft()` / `equip()` 把设置写进配置，`ESMoE` 接受 options 映射并按名解析目标函数；`scripts/train.py` 记录的块配置从训练完的模型上读，请求与实际不符即在开跑前退出。**用 0.1.4 的 `equip()` + `configure()` 设过这些开关的，训练出来的是构造函数默认值，请按 `scripts/blockspec.py` 复核。**
- **`dynamic_threshold` 无法追踪。** 掩码原先用 `scatter_` 塞一个 Python 布尔量，追踪器没有对应的算子，`torch.jit.trace` 与建立在它之上的导出全部失败。改成张量运算，逐输入重算。
- **路由统计混入了 warmup 前向。** 在加速卡上 ultralytics 会在第一个真实批次前跑一次空前向，路由钩子把那一行也收了进去——549 行对 548 张图。CPU 上不触发，所以此前一直没暴露。
- **同臂并行训练互相截断配置。** 嫁接出的 `configs/*.yaml` 原先不含 seed，两条并行车道跑同一条臂时写同一个文件，一个把另一个正在读的截断，读的那个死在 `KeyError: 'backbone'`。
- **`report.py` 跨硬件折算。** 分组键不含硬件，同一配置在两台机器上的运行被当成重复样本折进同一格。现在硬件栈与块配置都进入键。
- **`buckets.py` 在部分加速卡上无法评测。** `val()` 在推理模式里融合 conv+bn，有的构建拒绝对 inference tensor 取 view。现在提前融合，数值不变。
- **路由器 logits 未钳位。** 混合精度下跑飞的 logit 到 softmax 已是 inf，整个门控变 NaN。现按上游做法夹到 `[-30, 30]` 后走 fp32 softmax。

## 此前的 0.1.4

### 反馈与迭代

0.1.4 的多数条目来自首轮使用反馈：评测口径改用 COCO 式 32²/96² 分档与 maxDets=500（`scripts/buckets.py`，口径来源已在文档注明）；`--patience` 与 `IMGSZ` 是为对齐仓库复现协议（imgsz 800、120 epoch、patience 0）而加；半精度测试检查 FP32/AMP 下损失与梯度是否有限且一致。上游侧的反馈同样闭环：`OptimizedMOE` 追踪守卫的修复已被 YOLO-Master 合并（#241）。

### 修复

- `scripts/report.py` 的分组键补进 imgsz。此前同 epoch 不同分辨率的记录会被平均进同一行，而文档说过不会出现这种情况。

## 此前的 0.1.3

- **导出的模型不再忽略路由。** 块会跳过门控为零的专家，而这是一个依赖数据的判断：追踪器只记录示例输入走过的那条路由，导出的图便对此后所有输入沿用同一批专家。在一个路由随输入变化的块上，用某个输入导出的 ONNX 与 PyTorch 在另一条路由的输入上相差 0.2，现在相差 1e-7。追踪期间块会跑满所有专家，运行时仍走捷径，导出之外没有变慢。

    用 0.1.0 至 0.1.2 导出过模型的，请重新导出。

### 新增

- `scripts/verify.py`：单测做不到的正确性检查——真实训练一轮并确认辅助损失为正、`weight=0` 时损失表不变、检查点往返、断点续训、多个块一起训练、`val` 与 `predict`，以及 ONNX 导出。
- 一项回归测试：导出一个路由随输入符号变化的块，把两条分支都与 PyTorch 对照。

## 更早的小版本

0.1.0 是首个版本，给出 `inject_esmoe`、`graft`、`attach_aux_loss`、`collect_aux_loss` 四个入口，并附选型与三 seed 证据。0.1.1 修好 `equip()` 不带 `out` 时把配置当字典交给 `YOLO()` 的问题，补上 Colab 快速上手。0.1.2 修好 8.4.13x 之后 `loss_names` 在 `on_train_start` 时为空、导致训练日志表头错位的问题，并把包从 `src/` 移到仓库根目录。

## 安装

    pip install esmoe
