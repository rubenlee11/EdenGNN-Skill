---
name: edengnn
description: >-
  使用 EdenGNN 根据晶体或材料结构预测电子性质。
  当用户想用机器学习预测电荷密度（charge density、electron density）、想找通用模型来做电荷密度预测，或想根据材料结构预测电子能带、通过 ABACUS 和 PyATB 计算光学性质（介电函数、光学电导率）时，优先使用此技能。
  用户无需明确提到 EdenGNN.
---

# EdenGNN

## 选择流程

- **通用模型（默认）**：只读所选后端的 [vasp.md](references/vasp.md) 或 [abacus.md](references/abacus.md)，按顺序连续执行。用户同时指定两种后端才读两份。
- **训练或自训练权重**：再读 [training.md](references/training.md) 中对应后端部分；普通配置模板为 `~/.edengnn/config.yaml`。

- **VASP 与 DFT 对比或验证预测是否正确**：执行 [vasp.md](references/vasp.md) 的“DFT 对照与验证”。提前准备预测侧 `band/nscf` 和 target 侧 `scf/band_scf`，按 `stat.py` 的目录约定复用已有结果，通过命令行入口批量叠图和统计；即使只要求能带对比，也保留该入口所需的 SCF/NSCF 输入。

- **ABACUS 光学性质**：读 [abacus-optics.md](references/abacus-optics.md)，复用通用模型预测，输出 H/S/rR 后用 PyATB 计算介电函数和光学电导率。只请求光学时省略 band/DOS；已有矩阵可直接进入 PyATB。用户要求与 DFT 对比时，按该文档的“DFT 对比”生成收敛 SCF target（只输出 H/S，复用预测侧 rR），批量叠图并统计光谱误差。

## 执行边界

- 正常流程不读源码、不扫描仓库、不检查安装来源、不比对 checkpoint 内部结构、不做试运行或临时另写流程脚本；光学流程使用 `~/.edengnn/scripts/abacus/abacus_optics.py`。执行命令失败时先向用户反馈问题，而不是自行解决。
- 模板默认值直接保留。通用模型只改输出目录、输入列表和必要路径（展开 `~`、修正已知文件名）；chunk_size_train和chunk_size_predict分块参数需要根据gpu修改，80G显存分块20000000，按比例设置。
- 模型、config、DFT 模板默认位于（本地或服务器，看用户指定） `~/.edengnn/edengnn-uni/<后端>/`；默认文件可用就用，不重复询问。缺失时再请用户指定本地来源或确认下载来源。用户负责配置 pp.json 中的赝势/轨道路径，已有确认的配置直接复用；仅在缺失、无效时请求补齐，不替用户选版本。
- 默认预测完成后运行固定密度 band 和均匀网格 DOS，再用现成 `plot_band.py` 出图。用户明确只要 band 且不要求 DFT 对照时可省略 DOS；VASP 对照需保留统计入口所需的均匀网格 NSCF；只要光学时按光学流程执行；不做额外 SCF 对照、网格收敛、训练或精度评估，除非用户要求。
- 多结构绘图必须把所有结果目录写入同一个 `plot.txt`，只调用一次 `plot_band.py --filelist`；`--nproc` 使用作业模板或已分配资源允许的绘图并发数，不要在 shell 中逐结构调用绘图脚本。

## 运行环境与任务提交

- 在实际执行任务的机器（以下称为作业机器）上读取 `~/.edengnn/` 的脚本，模型权重以及作业模板。从作业模板中获取运行环境和可执行程序路径，若没有则询问用户作业提交模板。若作业模板没有指定提交脚本的资源，根据作业机器的运行方式、已分配资源和任务规模，自行生成提交脚本并提交，优先考虑最小化核时；当指定资源时，沿用作业模板的资源。不在服务器登录节点运行python脚本或者第一性计算。

- 除非用户指定只执行某一步，否则提前准备完整流程：本地将各阶段合并为一条连续执行命令；在集群上，如果是CPU/GPU混合运行，则按 CPU/GPU 资源拆分作业，一次提交带成功依赖的作业链（如 Slurm `afterok`、PBS `depend=afterok`），由调度器自动衔接，不等待 AI 在阶段结束后生成下一阶段命令；在集群上，如果是纯CPU运行，则合并成一个作业任务提交，减少等待python加载的时间。各作业使用 `set -e`，失败时阻止下游运行。任务运行时只等待或报告进度。独立后端或 band/DOS 在资源允许时并行；绘图命令成功退出就直接交付约定的产物路径。

- 通用模型作业链：VASP 为 CPU 准备/SAD → GPU density 与 augmentation 预测 → CPU band/DOS 与批量绘图；ABACUS 为 GPU 准备/预测 → CPU band/DOS 与批量绘图。阶段间文件写入节点共享的工作目录，下游在运行时解析实际预测时间戳。若 band/DOS 拆成多个作业，绘图须依赖全部计算成功后仅启动一次。CPU 与 GPU 作业分别申请对应分区和资源；依赖完成后仍需正常排队。

- 使用cpu训练或者推理的时候，num_gpus:不能设置成0，而是1，否则会报错。
