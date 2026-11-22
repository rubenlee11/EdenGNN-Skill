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

## 4. DFT 对照与验证（仅在用户要求时执行）

适用于通用模型和自训练权重。用户要求验证预测、与 DFT/target 对比或叠加能带时，提前把本节加入完整作业流程。即使只要求能带叠图，也执行第 3 节的预测 NSCF：现有 `stat.py` 命令行入口同时读取 SCF/NSCF 做统计，不能只提供两套能带，也不要自行导入内部绘图函数。

### 目录与复用

默认把同一结构的四类结果放在同一个结构目录中，`plot.txt` 每行填写该结构目录：

```text
$W/final/<name>/
  CHGCAR       # 模型预测的完整电荷密度（含 PAW augmentation）
  band/        # 预测 CHGCAR，ICHARG=11，KPOINTS_BAND
  nscf/        # 预测 CHGCAR，ICHARG=11，KPOINTS_SCF；也是预测 DOS
  scf/         # 收敛 DFT target，ICHARG=1，KPOINTS_SCF
  band_scf/    # target scf/CHGCAR，ICHARG=11，KPOINTS_BAND
```

`scf/` 和 `nscf/` 均需 `vasprun.xml`、`KPOINTS`、`CHGCAR`；`band/` 和 `band_scf/` 均需 `vasprun.xml`、`KPOINTS`。NSCF 使用 `LCHARG=False` 时，保留输入的预测 CHGCAR 供统计读取。

先复用已有计算：第 3 节的 `band/`、`nscf/` 无需重算；训练标签或此前 DFT 对照中的收敛 SCF 可通过复制或符号链接放到 `scf/`。复用前确认与预测侧的结构、原子顺序、PAW、ENCUT、PREC、FFT 网格和均匀 k 网格一致；SAD 的 `NELM=0` 结果不能作为 target SCF。已有匹配的 `band_scf/` 也直接复用。缺哪项才补算哪项。

若已有 target 位于独立目录，不必搬动：传入 `--dir-tar "$W/target"`，则脚本从 `$W/target/<name>/scf/` 和 `$W/target/<name>/band_scf/` 读取参考结果；预测侧仍为 `plot.txt` 中的 `$W/final/<name>/{band,nscf}/`。例如已有训练标签 `$W/scf/<name>/` 时，可将其链接为 `$W/final/<name>/scf`，不要把 SCF 父目录或 SAD 目录写进 `plot.txt`。

### 缺少 target 时补算

以下以默认同目录布局为例；多结构逐项准备和计算，最后统一调用统计。沿用本次 SAD 的 POSCAR、POTCAR、FFT 设置和配套 INCAR，不重新转换原胞或选择另一套赝势。

```bash
# 仅在没有可复用 target SCF 时执行。
mkdir -p "$W/final/$NAME/scf"
cp "$W/sad/$NAME/"{POSCAR,POTCAR,INCAR,CHGCAR} "$W/final/$NAME/scf/"
cp "$W/sad/$NAME/KPOINTS_SCF" "$W/final/$NAME/scf/KPOINTS"
python - <<'PY'
import os
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
path = Path(os.environ['W'])/'final'/os.environ['NAME']/'scf/INCAR'
cfg = Incar.from_file(path)
cfg.update(ICHARG=1, NELM=100, LCHARG=True, NEDOS=2001)
cfg.write_file(path)
PY
(cd "$W/final/$NAME/scf" && bash "$E/scripts/vasp/submit.sh")
```

SCF 必须电子收敛后才作为 target；未收敛则报告并停止后续对照，不能把正常退出等同于收敛。对新算或复用的 SCF，检查收敛并生成缺少的 target 能带输入：

```bash
python - <<'PY'
import os
import shutil
from pathlib import Path
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Vasprun
base = Path(os.environ['W'])/'final'/os.environ['NAME']
scf = Vasprun(base/'scf/vasprun.xml', parse_dos=False, parse_eigen=False,
              parse_projected_eigen=False, parse_potcar_file=False)
if not scf.converged_electronic:
    raise RuntimeError(f'Target SCF did not converge: {base / "scf"}')
band_dir = base/'band_scf'
band_dir.mkdir()
for filename in ('POSCAR', 'POTCAR', 'INCAR', 'CHGCAR'):
    shutil.copy2(base/'scf'/filename, band_dir/filename)
shutil.copy2(Path(os.environ['W'])/'sad'/os.environ['NAME']/'KPOINTS_BAND',
             band_dir/'KPOINTS')
cfg = Incar.from_file(band_dir/'INCAR')
cfg.update(ICHARG=11, NELM=100, LCHARG=False)
cfg.write_file(band_dir/'INCAR')
PY
(cd "$W/final/$NAME/band_scf" && bash "$E/scripts/vasp/submit.sh")
```

### 批量对比入口

全部结构计算完成后，使用第 3 节包含所有结构目录的同一份 `plot.txt`，只调用一次统计入口：

```bash
python "$E/scripts/vasp/stat.py" \
  --filelist "$W/plot.txt" --save-dir "$W/statistics" \
  --nproc "${plot_nproc:-${ncore:-1}}"
# target 分开放置时，在上述命令中加 --dir-tar "$W/target"。
```

该入口生成各结构的 `band_compare.png` 和 `$W/statistics/stat_mae.json`，包含密度、augmentation、本征值、能量、带隙和力的统计。能带叠图以 target 的 VBM（金属为 Ef）为共同零点；本征值 MAE 来自均匀 k 网格，不能称作高对称路径上的误差。第 3 节的 `band.png` 是预测能带/DOS 图，不能替代 `band_compare.png`。

直接运行现有脚本的命令行入口；不要写 `from stat import ...`、使用 importlib 加载脚本或另写比较绘图代码。命令报错时先反馈报错及涉及的输入目录，不临时绕过。交付时核对统计结果覆盖全部请求结构，不能仅因命令退出成功就把部分结构的结果当作全部完成。
