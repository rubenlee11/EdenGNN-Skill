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
