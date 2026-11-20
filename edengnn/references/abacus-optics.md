# ABACUS 通用模型 → PyATB 光学性质

只在用户请求光学性质时读本页。先按 [abacus.md](abacus.md) 第 1 节批量预测，保留实际时间戳目录 `T`；已有本次预测时直接复用。只请求光学性质可跳过 band/DOS，不额外 SCF；用户要求 DFT 对比时执行下方“DFT 对比”。使用用户已配置的赝势和轨道，eqgnn 环境中的 `abacus` 与 `pyatb`。

## 1. 固定预测密度输出矩阵

输入为 `T/predict/<name>/` 的完整计算目录，包括程序生成的 primitive `STRU`、赝势、轨道、KPT 和 `OUT.EDENGNN/SPIN1_CHG.cube`。不能用原始 CIF 晶格替代实际计算的 STRU 晶格。

在本次工作目录 `W` 中准备矩阵计算；`E=~/.edengnn`，`T` 沿用预测得到的实际时间戳：

```bash
set -e
mkdir -p "$W/calculations"
for source in "$T"/predict/*; do
  name=$(basename "$source")
  calc="$W/calculations/$name"
  cp -r "$source" "$calc"
  sed -i -E '/^(out_mat_hs2|out_mat_r|out_band|out_dos|symmetry)[[:space:]]/d' "$calc/INPUT"
  cat >> "$calc/INPUT" <<'EOF'
out_mat_hs2 1
out_mat_r 1
out_band 0
out_dos 0
symmetry -1
EOF
done
printf '%s\n' "$W"/calculations/* > "$W/optics.txt"
```

保留 `calculation nscf`、`basis_type lcao`、`init_chg drho` 和 `out_chg -1`。`out_mat_hs2 1` 输出 H/S，`out_mat_r 1` 输出位置矩阵，不能使用 gamma-only 局域轨道算法。保留生成的均匀 KPT；PyATB 积分网格在下一步独立设置。

按实际机器的 ABACUS 模板执行所有 `optics.txt` 中的目录。本地示例（`ncore` 根据资源、基组和结构规模选择）：

```bash
while IFS= read -r calc; do
  (cd "$calc" && bash "$E/scripts/abacus/submit.sh")
done < "$W/optics.txt"
```

集群直接打包准备好的计算目录传到 `~/.tmp`，修改现有 ABACUS 提交模板，申请合适核数提交；不在登录节点重新运行预测或 Python 准备。矩阵计算完成后整体同步回来，继续下节。不要把带 `#SBATCH` 的模板直接当本地脚本执行。

每个结构产物在 `calculations/<name>/OUT.EDENGNN/`：

- `data-HR-sparse_SPIN0.csr`：H，单位 Ry。
- `data-SR-sparse_SPIN0.csr`：S。
- `data-rR-sparse.csr`：位置矩阵，单位 Bohr。
- `running_nscf.log`：读取本次 `EFERMI`（eV）。

## 2. PyATB 计算并批量绘图

辅助脚本源码位于 `EdenGNN-main/scripts/abacus/abacus_optics.py`，安装时复制到 `~/.edengnn/scripts/abacus/abacus_optics.py`。它只做 Input 生成和批量绘图，不运行 DFT。它从实际 STRU 读取晶格、从本次日志读取费米能，并写入矩阵的绝对路径。`nspin` 取实际 INPUT（支持 1/4）；`occ_band -1` 按费米能判占据，金属不能简单取电子数一半作为固定占据带数。

```bash
conda activate eqgnn
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
# OMP/MKL 线程数和 MPI 进程数按实际已分配资源设置。
S="$E/scripts/abacus/abacus_optics.py"
cd "$W"
python "$S" prepare --filelist "$W/optics.txt" \
  --grid 12 12 12 --omega 0.1 10 --domega 0.05 --eta 0.1
while IFS= read -r calc; do
  (cd "$calc/pyatb" && pyatb > pyatb.log 2>&1)
done < "$W/optics.txt"
python "$S" plot --filelist "$W/optics.txt"
```

参数：`--grid` 是整个布里渊区的积分网格；`--omega`、`--domega` 是光子能量范围与步长（eV）；`--eta` 是展宽（eV）。上面是流程演示参数，不代表光谱已收敛；用户需要定量光谱时再按其要求加密。避开 omega=0 的介电函数奇点。`Input` 的 `max_kpoint_num 256` 控制内存分块，不改变积分网格。

### 光学 K 点密度与默认值

区分 ABACUS 的均匀 KPT（SCF 密度/费米能和矩阵计算）与 PyATB 的光学积分 `grid`。加密后者无需重新预测或重新计算已有 H/S/rR；不能用更密的 PyATB 网格代替 SCF 密度的 K 点收敛。

当前 eqgnn 安装的 PyATB 计算接口要求显式指定 `grid`、`omega`、`domega`；接口的 eta 默认 0.01 eV、method 默认 1。`pyatb_input --optical` 的生成器则按 `|b_i| / 0.05 Å⁻¹` 取整生成网格（倒格矢含 2π），并写入 omega=0–10 eV、domega=0.01 eV、eta=0.1 eV。这里的 12³ 和 0.05 eV 步长是快速演示设置，不是 PyATB 默认值。当前安装生成器的 optical 分支使用固定 0.05 间距，不能假定 `--mp` 会修改它。

定量光谱可以该自动网格为起点，保持 eta 和能量范围固定，进一步加密后比较峰位与强度；小晶胞常需数十点每方向，金属、窄展宽或尖锐结构可能需要更密。不存在对所有材料都收敛的固定网格。不自动做整套收敛扫描，按用户所需精度决定。当前四个测试 primitive cell 对应生成器网格：Al 53³、GaAs 38³、Si 40³、ZnO 47³。两侧对比必须使用相同的积分网格。

当前计算是独立粒子带间光学响应，不含激子、声子辅助跃迁；金属完整低频响应还需 Drude 项，不能把本页带间结果当作完整金属介电函数。

## 3. 交付

每个结构报告以下实际绝对路径以及原始预测时间戳 `T`：

- `calculations/<name>/pyatb/Input`：PyATB 配置。
- `calculations/<name>/pyatb/Out/Optical_Conductivity/`：`dielectric_function_{real,imag}_part.dat`（无量纲），`optical_conductivity_{real,imag}_part.dat`（S/m）。第一列为 eV，后面依次为 xx, xy, xz, yx, yy, yz, zx, zy, zz。
- `calculations/<name>/dielectric_function_{xx,yy,zz}.png`：三个方向分开，每张图同时绘制介电函数实部、虚部。
- `calculations/<name>/optical_conductivity_{xx,yy,zz}.png`：三个方向分开，只绘制光学电导率实部。共六张独立正方形图（6×6 英寸），不拼接面板，横坐标显示 0–6 eV；原始光谱和误差统计仍保留完整计算能量范围。
- `calculations/<name>/pyatb/pyatb.log`、`pyatb/Out/running.log`：计算日志。

绘图一次传入完整 filelist；命令成功后直接交付，不另开图片检查。已有 H/S/rR 时只执行第 2 节，无需重新预测或跑 ABACUS。

## DFT 对比

用户请求通用 ABACUS 模型与 DFT 光学性质对比时，预测一侧复用上面完成的 `calculations/<name>`，不重新预测。target 使用同一 ABACUS 的收敛 SCF 密度输出 H/S，rR 复用预测侧；不能把另一次固定预测密度的 NSCF 标为 DFT target。

两侧使用完全相同的实际 primitive STRU、赝势、轨道、泛函、ecutwfc、KPT、展宽、自旋/SOC 设置。target 设置 `calculation scf`、`init_chg atomic`、`out_mat_hs2 1`、`out_mat_r 0`。相同结构、原子顺序、轨道基组及自旋表示下，rR 不依赖电荷密度，两个分支只计算一次；若这些设置不同则不能复用。已有匹配的收敛 SCF 和矩阵时直接复用，不重算；已有收敛密度但没有矩阵时，以 `init_chg file` 做固定 DFT 密度的 NSCF 输出矩阵，不能把 total density 当作 drho。

新 target 示例，从预测计算目录只复制输入，不复制预测密度或旧输出：

```bash
set -e
mkdir -p "$W/target"
: > "$W/target_optics.txt"
while IFS= read -r prediction; do
  name=$(basename "$prediction")
  target="$W/target/$name"
  mkdir -p "$target"
  cp "$prediction/INPUT" "$prediction/STRU" "$prediction/KPT" \
     "$prediction/"*.upf "$prediction/"*.orb "$target/"
  sed -i -E '/^(calculation|init_chg|out_chg|out_mat_r)[[:space:]]/d' "$target/INPUT"
  cat >> "$target/INPUT" <<'EOF'
calculation scf
init_chg atomic
out_chg 1
out_mat_r 0
EOF
  printf '%s\n' "$target" >> "$W/target_optics.txt"
done < "$W/optics.txt"
```

如果赝势/轨道由 INPUT 的共享目录引用，沿用该设置，无需上述 `*.upf/*.orb` 复制。目标目录使用新的空目录，避免残留文件。按实际 ABACUS 提交模板批量运行 `target_optics.txt`，集群做法同第 1 节；确认日志报告 SCF 收敛后再计算 target 光谱，未收敛不能用于对比。

PyATB 两侧使用相同的 grid、omega、domega、eta、method 和自旋设置，但各自读取对应计算的费米能；不做人工能量平移或剪刀修正。对绝缘体/金属都用 `occ_band -1`。target 的命令（DFT 结束且同步回实际 PyATB 运行机器后，建立指向预测侧 rR 的绝对路径软链接；以下对应上述 EDENGNN 后缀和目录布局）：

```bash
while IFS= read -r prediction; do
  name=$(basename "$prediction")
  output="$W/target/$name/OUT.EDENGNN"
  mkdir -p "$output"
  ln -sfn "$(realpath "$prediction/OUT.EDENGNN/data-rR-sparse.csr")" \
    "$output/data-rR-sparse.csr"
done < "$W/optics.txt"
python "$S" prepare --filelist "$W/target_optics.txt" \
  --grid 12 12 12 --omega 0.1 10 --domega 0.05 --eta 0.1
while IFS= read -r calc; do
  (cd "$calc/pyatb" && pyatb > pyatb.log 2>&1)
done < "$W/target_optics.txt"
python "$S" compare --filelist "$W/optics.txt" \
  --target-filelist "$W/target_optics.txt"
```

`--filelist` 与 `--target-filelist` 每行一一对应同一结构，列表可以指向不同工作目录，不能靠两次 glob 的偶然顺序配对。预测和 target 均已计算时只调用一次 `compare`，不再执行上游步骤。

每个预测目录新增：

- `dielectric_function_{xx,yy,zz}_compare.png`、`optical_conductivity_{xx,yy,zz}_compare.png`：六张独立对比图；每张只画一个方向。介电函数实部、虚部同图（蓝/橙），电导率只画实部；实线预测、虚线 DFT。
- `optical_compare.json`：记录两侧实际路径、能量范围、每项光谱 xx/yy/zz 的 MAE 和完整九分量张量 MAE；介电函数无量纲，光学电导率单位 S/m。不归一化、不插值、不平移。

交付对比图、误差文件、DFT target 路径、原始预测时间戳及光谱参数。误差仅代表相同独立粒子设置下的预测与 DFT 差异。

## 验证范围

2026-11-19 已实跑通用 ABACUS 模型的 Al、Si、GaAs、ZnO：本地预测 → lxw 固定预测密度输出 H/S/rR → 本地 eqgnn PyATB → 批量绘图。测试使用 nspin=1、12×12×12 网格、0.1–10 eV、步长 0.05 eV、展宽 0.1 eV；结果位于 `aips_tutorial/results/abacus_optics/`。nspin=4 的光学路径依据 PyATB 接口整理，本次未实跑；VASP 和自训练权重的光学流程尚未验证。

DFT 对比已于 2026-11-19 对 Al、Si、GaAs、ZnO 完整实跑：lxw 收敛 SCF → 回传 H/S/rR → eqgnn PyATB → 批量对比图和 MAE。四者日志均确认 charge density convergence is achieved。target 在 `aips_tutorial/results/abacus_optics/target/`，对比图和 JSON 在对应 `calculations/<name>/`。两侧使用相同 12³ 演示积分网格，未做网格收敛。

DFT target 的 rR 复用：已确认上述四结构预测/DFT 的 rR 逐字节相同；优化后的流程仅在预测侧输出 rR，DFT 侧通过软链接复用。
