# 普通训练与自训练预测

仅在用户要求训练或使用自己训练的权重时读取。使用`~/.edengnn/config.yaml`的默认配置文件，复制到工作目录修改。
以下 `$E=$HOME/.edengnn`，`$W` 为本次源码目录外工作目录，路径写成绝对路径。
环境从 `$E/cpu.sh`、`$E/cuda.sh` 获取；DFT 运行命令调用 `$E/scripts/<vasp或abacus>/submit.sh`。沿用模板或用户已有授权的 ncore 和并发策略；确实缺少时才询问，不自行写死路径或核数。

## 演示训练

- 当用户只给了很少的训练结构，比如少于10个，或者说是演示训练时，train.txt通过重复的方式增加到50个，这样能减少等待的时间。

- 演示训练时Scheduler设置，用CosineAnnealingLR，max_steps和T_max设置为100步。

## VASP 训练

输入：用户结构列表、已配置的 PAW 赝势映射文件和匹配训练精度的 INCAR。使用普通模板 `$E/scripts/vasp/INCAR` 和 `$E/scripts/vasp/pp.json`（用户可指定其他文件）。数据准备仍按 [vasp.md](vasp.md) 的 SAD 节执行，将其中 `--path-input`、`--path-pp` 换成这些普通文件。

SAD、SCF 的结构、原子顺序、FFT 设置与 PAW 保持一致。生成标签：

```bash
cp -r "$W/sad" "$W/scf"
python - <<'PY'
import os, shutil
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
for path in (Path(os.environ['W'])/'scf').iterdir():
    shutil.copy2(path/'KPOINTS_SCF', path/'KPOINTS')
    cfg = Incar.from_file(path/'INCAR')
    cfg.update(ICHARG=1, NELM=100, LCHARG=True)
    cfg.write_file(path/'INCAR')
PY
for path in "$W/scf/"*; do
  (cd "$path" && bash "$E/scripts/vasp/submit.sh")
done
cp "$E/config.yaml" "$W/train.yaml"
```

按结构划分 train.txt/val.txt，每行是 SAD 目录。`data.vasp.dir` 指向 SCF 父目录，通过同名子目录匹配标签。修改字段：

| 字段 | 值 |
| --- | --- |
| run.mode / task | train / 2（联合），或 0（density）、1（augmentation） |
| run.save_dir / checkpoint / resume | `$W/train` / null / false |
| data.dft_software | vasp |
| data.vasp.dir / path_pp / lmix_max | `$W/scf` / 用户映射 / 与 INCAR 一致，通常 6 |
| data.path_train / path_val | 两个实际划分列表 |
| data.use_bin / batch_size | false / 1 |
| model.use_sad | task=0/2 为 true；task=1 为 false |
| optimize.loss_dict.grid_func_out.weight_train、weight_val | 联合可用 0.9，单 density 为 1，单 aug 为 0 |
| optimize.loss_dict.aug_tensor.weight_train、weight_val | 联合可用 0.1，单 aug 为 1，单 density 为 0 |

保留 `model.task=${run.task}`、`optimize.task=${run.task}`、`model.lmix_max=${data.vasp.lmix_max}`，其他参数按任务需要改。当前总密度标签配合 use_sad=true 学习差分；已有模型预测不得随意切换其训练时的 use_sad。

```bash
edengnn-train --config "$W/train.yaml" > "$W/train.log" 2>&1
```

产物 `$W/train/edengnn/<timestamp>/{best_step=<step>.ckpt,info.log,wandb/}`。训练需有验证周期并保存 checkpoint；`best_step` 从第二轮才开始保存，单轮可用 `checkpoints/epoch=0-step=1.ckpt`（普通 checkpoint），短流程从头训练至少两轮才按 best 路径取权重；单结构演示不代表泛化验证。

## VASP 自训练权重预测

输入：新结构、训练配置、实际 best checkpoint、相同 PAW/INCAR 设置。为新结构运行 SAD，创建含 SAD 目录的 predict.txt。

```bash
cp "$E/config.yaml" "$W/self_predict.yaml"
```

在新模板中沿用训练的 `model`、`run.task`、`run.dtype`、`data.vasp.lmix_max` 和赝势映射；设置 `run.checkpoint`、`run.save_dir=$W/self_predict`、`data.dft_software=vasp`、`data.path_predict=$W/predict.txt`、`run.mode=predict`、`run.resume=false`、`data.use_bin=false`。

```bash
edengnn-predict --config "$W/self_predict.yaml" > "$W/self_predict.log" 2>&1
```

产物 `$W/self_predict/edengnn/<timestamp>/predict/<name>`。task=2 输出已含 density 和 augmentation，直接当 CHGCAR 使用；分开训练的 task=0/1 各预测一次，density 在前拼接。接 [vasp.md](vasp.md) 第 3 节执行 band/DOS：task=2 将本次预测文件复制为 `$W/final/<name>/CHGCAR`，跳过通用模型的双时间戳选择和 cat；分开训练则将 T_D/T_A 设为这次实际输出。只有 density 分支不等于完整 PAW 预测。

## ABACUS 训练

输入：用户结构列表、配好的赝势/轨道映射和无头 INPUT 模板。普通模板用 `$E/scripts/abacus/INPUT`，保持训练要求的 cutoff/轨道设置。补丁版 ABACUS 的 out_chg=3 输出差分标签。

```bash
python "$E/scripts/abacus/gen_abacus_cal.py" \
  --filelist "$W/structures.txt" --save-dir "$W/scf" \
  --path-pp "$E/scripts/abacus/pp.json" --path-input "$E/scripts/abacus/INPUT" \
  --for-band --nproc 1
for path in "$W/scf/"*; do
  (cd "$path" && bash "$E/scripts/abacus/submit.sh")
done
cp "$E/config.yaml" "$W/train.yaml"
```

生成器设置 scf/lcao/atomic/out_chg=3，输出 `scf/<name>/OUT.<suffix>/chgdelta.cube`。训练列表按结构划分，每行是 SCF **目录**。修改：

| 字段 | 值 |
| --- | --- |
| run.mode / task / checkpoint / resume | train / 0 / null / false |
| run.save_dir | `$W/train` |
| data.dft_software / abacus.prefix | abacus / 实际 suffix |
| data.abacus.path_pp / path_input | 用户映射 / 匹配训练 cutoff 的普通 INPUT |
| data.path_train / path_val | 含 SCF 目录的两个划分列表 |
| data.batch_size / model.use_sad | 1 / false |
| optimize.loss_dict.grid_func_out.weight_train、weight_val | 1 |
| optimize.loss_dict.aug_tensor.weight_train、weight_val | 0 |

```bash
edengnn-train --config "$W/train.yaml" > "$W/train.log" 2>&1
```

产物 `$W/train/edengnn/<timestamp>/{best_step=<step>.ckpt,info.log,wandb/}`。差分标签不能用总密度 cube 代替。

## ABACUS 自训练权重预测

输入：新结构列表（直接填 CIF/POSCAR）、训练配置、实际 checkpoint、相同赝势/轨道/ecutwfc。

```bash
cp "$E/config.yaml" "$W/self_predict.yaml"
```

沿用训练的 `model`、task=0、dtype 和 DFT 设置；修改 `run.mode=predict`、`run.resume=false`、`run.checkpoint`、`run.save_dir=$W/self_predict`、`data.dft_software=abacus`、`data.path_predict=$W/structures.txt`、`data.abacus.path_pp/path_input`，设 `data.plot_band=true`、`model.use_sad=false`。

```bash
edengnn-predict --config "$W/self_predict.yaml" > "$W/self_predict.log" 2>&1
```

产物 `$W/self_predict/edengnn/<timestamp>/predict/<name>/`，其中 `OUT.EDENGNN/SPIN1_CHG.cube` 为差分密度，suffix 固定为 EDENGNN。接 [abacus.md](abacus.md) 第 2 节，跳过通用模型 results 下的时间戳选择，用本次 self_predict 下实际时间戳设置 T，执行 band/DOS 和绘图。
