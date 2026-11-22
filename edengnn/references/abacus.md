# ABACUS 通用模型：结构 → 预测 → band/DOS 图

光学性质请求在完成第 1 节预测后转到 [abacus-optics.md](abacus-optics.md)，不执行下面的 band/DOS 步骤，除非用户同时要求。

不需要 SAD 或预先 SCF。使用 bash，替换 `STRUCTURE` 和本次新工作目录 `W`；已配置的 pp.json 直接使用。下面命令连续执行，无需读取源码。

```bash
set -e
export E="$HOME/.edengnn"
source "$E/cpu.sh"
source "$E/cuda.sh"
# ncore 和并发策略取自用户作业模板或已有授权；不在此固定。
export ncore
export PYTHONDONTWRITEBYTECODE=1
export W=/absolute/path/to/new/abacus_run
STRUCTURE='<用户提供的结构文件绝对路径>'
NAME=$(basename "$STRUCTURE")
export NAME="${NAME%.*}"
mkdir -p "$W"
printf '%s\n' "$STRUCTURE" > "$W/structures.txt"
```

## 1. 预测：只改五项

使用 `edengnn-uni/abacus/` 的配套 config、checkpoint、INPUT 和 pp.json。保留模板的 task、模型、采样和计算默认值，只设输出/列表并将路径展开为绝对路径。

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
model_dir = Path(os.environ['E'])/'edengnn-uni/abacus'
w = Path(os.environ['W'])
cfg = OmegaConf.load(model_dir/'config_abacus.yaml')
cfg.run.save_dir = str(w/'results')
cfg.run.checkpoint = str(model_dir/'EdenGNN-Uni_abacus.ckpt')
cfg.data.path_predict = str(w/'structures.txt')
cfg.data.abacus.path_pp = str(model_dir/'pp.json')
cfg.data.abacus.path_input = str(model_dir/'INPUT')
OmegaConf.save(cfg, w/'predict.yaml')
PY
edengnn-predict --config "$W/predict.yaml" > "$W/predict.log" 2>&1
```

预测直接写出计算输入 `INPUT,STRU,KPT,KPT_BAND`、赝势/轨道和差分密度 `OUT.EDENGNN/SPIN1_CHG.cube`

## 2. 计算、绘图

按作业模板执行；只有模板或用户允许时才并发。新 W 只产生一个预测时间戳，直接取本次路径；保留程序生成的文件，只切换 band 的 KPT 和两个 out 开关。

```bash
runs=("$W"/results/edengnn/*)
T=${runs[0]}
mkdir -p "$W/final/$NAME"
cp -r "$T/predict/$NAME" "$W/final/$NAME/band"
cp -r "$T/predict/$NAME" "$W/final/$NAME/nscf"
cp "$W/final/$NAME/band/KPT_BAND" "$W/final/$NAME/band/KPT"
python - <<'PY'
import os
from pathlib import Path
base = Path(os.environ['W'])/'final'/os.environ['NAME']
for mode in ('band', 'nscf'):
    path = base/mode/'INPUT'
    fields = dict(line.split(maxsplit=1) for line in path.read_text().splitlines()[1:] if line.strip())
    fields['out_band'] = '1' if mode == 'band' else '0'
    fields['out_dos'] = '0' if mode == 'band' else '2'
    path.write_text('INPUT_PARAMETERS\n' + ''.join(f'{key} {value}\n' for key, value in fields.items()))
PY
for mode in band nscf; do
  (cd "$W/final/$NAME/$mode" && bash "$E/scripts/abacus/submit.sh")
done
printf '%s\n' "$W/final/$NAME" > "$W/plot.txt"
# plot_nproc 取作业模板或已分配资源允许的绘图并发数；未提供时沿用 ncore，再无授权资源才用 1。
python "$E/scripts/abacus/plot_band.py" --filelist "$W/plot.txt" --nproc "${plot_nproc:-${ncore:-1}}"
printf '%s\n' "$T" "$W/final/$NAME/band.png"
```

输出 `final/<name>/band/OUT.EDENGNN/BANDS_1.dat`、`nscf/OUT.EDENGNN/{TDOS,PDOS}` 和 `band.png`。绘图脚本直接读 TDOS，自动叠加存在的 PDOS，不复制为 TDOS.dat，不另写画图程序。绘图命令成功退出后直接报告上述图像路径和时间戳，不检查 band.png、不打开图片；仅失败时读对应 `.log`/`abacus.out`。多结构逐项沿用同一流程，不另写执行脚本。

## DFT 能带对比的目录约定

仅在用户要求与 DFT 对比时，按现有 `scripts/abacus/stat.py` 的约定组织每个结构目录：

| 子目录 | 计算及用途 | 必需输出（位于 `OUT.<suffix>/`） |
| --- | --- | --- |
| `nscf/` | 固定模型预测密度，均匀 k 网格 NSCF | `BANDS_1.dat`、`running_nscf.log` |
| `band/` | 固定模型预测密度，高对称路径 NSCF | `BANDS_1.dat`、`running_nscf.log` |
| `scf/` | DFT 参考，均匀 k 网格的实际收敛 SCF | `BANDS_1.dat`、`running_scf.log` |
| `band_scf/` | 固定上述 DFT 收敛密度，高对称路径 NSCF | `BANDS_1.dat`、`running_nscf.log` |

两侧保持相同的结构、原子顺序、赝势、轨道、截断能和自旋设置；`nscf/` 与 `scf/` 的均匀 k 网格一致，`band/` 与 `band_scf/` 的路径一致。用于统计的计算均设 `out_band=1`，包括默认 DOS 流程中原设为 `out_band=0` 的 `nscf/`。

已有匹配的收敛 SCF 时，可复用其完整计算目录作为 `scf/`；不要将读取 DFT 密度的 NSCF 计算放入 `scf/`，也不要重命名日志或修改统计脚本来适配这种目录。`band_scf/` 才是使用 DFT 收敛密度的路径 NSCF 目录；读取总密度用 `init_chg=file`，读取差分密度用补丁版的 `init_chg=drho`，两者不能混用。

预测和参考目录默认同处 `final/<name>/`；参考单独存放时，使用 `--dir-tar <root>`，该根目录下须有 `<name>/{scf,band_scf}`。预测侧输出后缀由 `--prefix` 指定，参考侧由 `--prefix-tar` 指定，均须与各自实际 `OUT.<suffix>` 一致。
