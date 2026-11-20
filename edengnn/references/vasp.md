# VASP 通用模型：结构 → SAD → 预测 → band/DOS 图

使用 bash，先替换 `STRUCTURE` 和本次新工作目录 `W`；已配置的 pp.json 直接使用。以下命令按顺序连续执行，无需预读程序源码。所有中间产物保存在 `$W`。

```bash
set -e
export E="$HOME/.edengnn"
source "$E/cpu.sh"
source "$E/cuda.sh"
# ncore 和并发策略取自用户作业模板或已有授权；不在此固定。
export ncore
export PYTHONDONTWRITEBYTECODE=1
export W=/absolute/path/to/new/vasp_run
STRUCTURE='<用户提供的结构文件绝对路径>'
NAME=$(basename "$STRUCTURE")
export NAME="${NAME%.*}"
mkdir -p "$W"
printf '%s\n' "$STRUCTURE" > "$W/structures.txt"
```

## 1. SAD

模板和赝势映射均使用通用模型配套文件。生成输入的脚本不运行 VASP，必须接着运行 SAD：

```bash
python "$E/scripts/vasp/vasp_sad.py" \
  --filelist "$W/structures.txt" --save-dir "$W/sad" \
  --path-pp "$E/edengnn-uni/vasp/pp.json" \
  --path-input "$E/edengnn-uni/vasp/INCAR" \
  --for-scf --for-band --nproc 1
(cd "$W/sad/$NAME" && bash "$E/scripts/vasp/submit.sh")
printf '%s\n' "$W/sad/$NAME" > "$W/predict.txt"
```

生成的 POSCAR 是后续统一使用的原胞。`ICHARG=12,NELM=0` 由脚本设置；输出 `sad/<name>/CHGCAR`、`KPOINTS_SCF`、`KPOINTS_BAND`。预测列表填 **SAD 目录**。

## 2. 预测：只改四项

从配套完整 config 生成本次 YAML。其余 task、模型、SAD 开关设置直接保留。

```bash
python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
model_dir = Path(os.environ['E'])/'edengnn-uni/vasp'
w = Path(os.environ['W'])
for kind, weight in [('density', 'pseudo'), ('aug', 'aug')]:
    cfg = OmegaConf.load(model_dir/f'config_vasp_{kind}.yaml')
    cfg.run.save_dir = str(w/kind)
    cfg.run.checkpoint = str(model_dir/f'EdenGNN-Uni_vasp_{weight}.ckpt')
    cfg.data.path_predict = str(w/'predict.txt')
    cfg.data.vasp.path_pp = str(model_dir/'pp.json')
    OmegaConf.save(cfg, w/f'{kind}.yaml')
PY
edengnn-predict --config "$W/density.yaml" > "$W/density.log" 2>&1
edengnn-predict --config "$W/aug.yaml" > "$W/aug.log" 2>&1
```

checkpoint 路径显式指定是为展开 `~`，同时修正配套 augmentation config 的 `EdenGNN-Uni_aug.ckpt` 为实际 `EdenGNN-Uni_vasp_aug.ckpt`。预测输出是 `$T/predict/<name>` **文件**。

## 3. 拼接、计算、绘图

本次 W 是新目录，每种预测只有一个时间戳，直接取本次实际路径，不用反复搜索日志。

```bash
density_runs=("$W"/density/edengnn/*)
aug_runs=("$W"/aug/edengnn/*)
T_D=${density_runs[0]}
T_A=${aug_runs[0]}
mkdir -p "$W/final/$NAME"
cat "$T_D/predict/$NAME" "$T_A/predict/$NAME" > "$W/final/$NAME/CHGCAR"
for mode in band nscf; do
  mkdir -p "$W/final/$NAME/$mode"
  cp "$W/sad/$NAME/"{POSCAR,POTCAR,INCAR} "$W/final/$NAME/$mode/"
  cp "$W/final/$NAME/CHGCAR" "$W/final/$NAME/$mode/CHGCAR"
done
cp "$W/sad/$NAME/KPOINTS_BAND" "$W/final/$NAME/band/KPOINTS"
cp "$W/sad/$NAME/KPOINTS_SCF" "$W/final/$NAME/nscf/KPOINTS"
python - <<'PY'
import os
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
base = Path(os.environ['W'])/'final'/os.environ['NAME']
for mode in ('band', 'nscf'):
    path = base/mode/'INCAR'
    cfg = Incar.from_file(path)
    cfg.update(ICHARG=11, NELM=100, LCHARG=False, NEDOS=2001)
    cfg.write_file(path)
PY
for mode in band nscf; do
  (cd "$W/final/$NAME/$mode" && bash "$E/scripts/vasp/submit.sh")
done
printf '%s\n' "$W/final/$NAME" > "$W/plot.txt"
# plot_nproc 取作业模板或已分配资源允许的绘图并发数；未提供时沿用 ncore，再无授权资源才用 1。
python "$E/scripts/vasp/plot_band.py" --filelist "$W/plot.txt" --nproc "${plot_nproc:-${ncore:-1}}"
printf '%s\n' "$T_D" "$T_A" "$W/final/$NAME/band.png"
```

density 在前、augmentation 在后；不要重复拼同一文件。固定密度计算必须将 SAD 的 NELM=0 改为正值。DOS 用均匀 KPOINTS_SCF，不用 band 路径。输出 `final/<name>/band/{vasprun.xml,EIGENVAL}`、`nscf/{vasprun.xml,DOSCAR}` 和 `band.png`。绘图命令成功退出后直接报告上述图像路径和时间戳，不检查 band.png、不打开图片；仅失败时读对应 `.log`/`vasp.out`。若结构名无扩展名，NAME 就是其文件名；多结构逐项沿用同一流程，不另写执行脚本。
