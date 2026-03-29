# Changelog

## 2026-03-26

这次改动把 binding regime 的初始化逻辑改成了“先 graph 传播，再只对高 nullity 候选批量 `_calc_nullity`”。

### 1. `find_all_regimes!` 现在的顺序

现在 binding 侧初始化流程变成：

1. `_enumerate_all_regimes(model._L_helper)` 枚举所有 perm
2. 立刻用 `all_perms + model._L_helper` 构造 x-neighbor regime graph
3. 构造 `BindRegime`，此时 `nullity` 先置为未知
4. 以 graph connected component 为单位，从 seed regime 出发传播 `H/H0`
5. 传播过程中直接把能判定的 regime 标成 `nullity = 0/1`
6. 只把传播中识别出的 `nullity >= 2` 候选 perm 收集起来，最后再批量 `_calc_nullity`

这样 `_calc_nullity` 不再是 binding 初始化的全量前置步骤，而是 deferred high-nullity fallback。

### 2. x-neighbor graph 的边改成更紧凑的存储

`VertexEdge` 以前直接存：

- `change_dir_x::SparseVector`

现在改成只存：

- `x_pos`
- `x_neg`
- `x_dim`
- `intersect_x`

也就是“哪一列系数是 `+1`、哪一列系数是 `-1`”，需要对外返回 `change_dir_x` 时再即时 materialize 成 sparse vector。

这样做的直接收益是：

- graph 边对象更轻
- graph 构造时少做很多 `SparseVector` 分配
- rank-1 传播时可以直接从边上拿 `j_from/j_to`，不用再 `findnz(change_dir_x)`

### 3. `nullity = 0/1` 的 affine info 现在由 graph 传播直接填出

- 从 regular source 沿边传播时：
  - `δ ≠ 0`，目标 regime 直接得到 regular `H/H0`
  - `δ = 0`，目标 regime 直接得到 nullity-1 的 rank-1 singular ray `H/H0`
- 对于 graph 中没有被 regular propagation 吃到的残余节点，再用 direct seed classification 补一轮
- 仍然不能在传播里解决的，才进入 deferred high-nullity 集合

这一步仍由 `_prefill_affine_cache!` 统一负责，但它现在不再依赖“先有全量 nullity”。

### 4. qK interface 计算复用同一套传播结果

`_fulfill_regimes_graph!` 不再对每个 low-nullity regime 再单独触发一次 `get_regime(...; inv_info=true)`，
而是直接调用 `_prefill_affine_cache!`，让 qK 接口构造复用同一套预填结果。

### 5. 文档与测试

- `Archetecture.md` 已更新初始化流水线
- `test/runtests.jl` 新增断言：`find_all_regimes!` 后 graph 已存在，且所有 `nullity <= 1` 的 regime 已有 `H/H0`

## 2026-03-24

这次改动修正了一个重要的 catalysis 侧数学假设错误：以前代码默认 `S` 的 regime 只是从 `v` 里“选项”，于是把 catalysis regime 的常数截距全部当成了 0。现在已经改成支持一般正线性映射 `S^+ v` / `S^- v`。

### 1. `CatalysisRegime` 现在保留 catalysis offset

新增并贯通了这些字段：

- `P0_pos_neg`
- `P0`
- `C0`

对应数学上：

```math
f\!\left(\begin{bmatrix}S^+\\S^-\end{bmatrix}\right)
=
\left(
\begin{bmatrix}P_0^+\\P_0^-\end{bmatrix},
\begin{bmatrix}P^+\\P^-\end{bmatrix},
\begin{bmatrix}C_0^+\\C_0^-\end{bmatrix},
\begin{bmatrix}C^+\\C^-\end{bmatrix}
\right).
```

所以现在代码里的 catalysis steady-state / dominance 是：

```math
P^\theta \Pi \log x + P^\theta \log k + P_0^\theta = 0,
```

```math
C^\theta \Pi \log x + C^\theta \log k + C_0^\theta \ge 0.
```

相关文件：

- `src/initialize.jl`
- `src/Catalysis_regime.jl`

### 2. Mixed regime 的 `(q,K,k)` 与 `(q_ss,K,k)` 条件都改成带截距版本

以下路径现在都会正确带上 catalysis offset：

- `get_C_C0_nullity_xk(rgm::BncRegime, :combined)`
- `_calc_C_qKk_catalysis_only_*`
- `_calc_C_qKk_cat_*`
- `_calc_C_qKk_ss_*`

也就是说：

- `(x,k)` 基底下 combined condition 现在含 `P0^θ` / `C0^θ`
- `(q,K,k)` 基底下 catalytic-only / combined consistency 现在含 `C0^θ`
- singular elimination 分支也不再默认这些常数项为 0

相关文件：

- `src/Bnc_regime.jl`

### 3. `K_ss` 展开回 `(K,k)` 时，`BncRegime.H0` 现在会吸收 `P0^θ`

以前代码只做了：

```math
\log K_{ss} = \begin{bmatrix}\log K\\-P^\theta \log k\end{bmatrix}
```

但正确形式应为：

```math
\log K_{ss} = \begin{bmatrix}\log K\\-(P^\theta \log k + P_0^\theta)\end{bmatrix}.
```

因此如果

```math
\log x = H_{ss}\log(q_{ss},K_{ss}) + H_{0,ss},
\qquad
H_{ss} = [H_L \;\; H_R],
```

则展开到 `(q_ss,K,k)` 后应为：

```math
H_{ssk} = [H_L \;\; -H_R P^\theta],
\qquad
H_{0,ssk} = H_{0,ss} - H_R P_0^\theta.
```

这一步已经写进 `_expand_Hss_to_qssKk`，并且 `BncRegime.H0` 现在保存的是展开后的 `H0_ssk`。

直接受影响的输出有：

- `get_H_H0(rgm::BncRegime)`
- `show_expression_x(rgm::BncRegime)`
- `get_qcat_F_F0(rgm::BncRegime)`
- `show_expression_qcat(rgm::BncRegime)`

相关文件：

- `src/Bnc_regime.jl`

### 4. Symbolics 渲染与 getter 现在会把 offset 显示出来

修正后：

- `show_condition_xk(cat_rgm; kind=:steady_state)` 使用 `P0^θ`
- `show_condition_xk(cat_rgm; kind=:dominance)` 使用 `C0^θ`
- mixed regime 的 `show_condition_xk / qKk / qssKk` 都会通过更新后的 getter 自动显示常数项

另外新增：

- `get_P0_pos_neg`

相关文件：

- `src/Catalysis_regime.jl`
- `src/symbolics.jl`
- `src/BindingAndCatalysis.jl`

### 5. 新增一个真正带 offset 的回归测试模型

测试里增加了：

```julia
Γ = [2 1 -1]
Π = [1 0 0; 0 1 0; 0 0 1]
```

这个例子会产生：

- 2 个 catalysis regimes
- 非零 `P0^θ`
- 非零 `C0^θ`

并且测试了：

- `CatalysisRegime` 的 `P0/C0`
- `BncRegime.H0` 的 `P0^θ` 展开
- `(q,K,k)` catalytic-only consistency 的常数项

相关文件：

- `test/runtests.jl`

### 6. 文档更新

新增或更新了：

- `Archetecture.md`
- `Matrix_relation.md`

这两份文档现在都明确写了：

- `S^+v` / `S^-v` 是正线性映射，不一定是 selector
- `CatalysisRegime` 为什么会带 `P0^θ` / `C0^θ`
- `K_ss`、`H_ssk`、`H0_ssk` 的正确关系


## 2026-03-23

这次改动的目标是减少 regime 构建过程中的重复计算与重复存储，同时尽量不破坏现有公共接口。

### 1. Binding regime 的 `H/H0` 改成“先共享 affine info，再按需实例化 qK 条件”

以前 `get_regime(...; inv_info=true)` 会把下面几件事绑在一起做：

- 计算 `H`
- 计算 `H0`
- 立刻实例化 `C_qK`, `C0_qK`

现在拆成了两层：

- affine info：`H`, `H0`
- qK condition：`C_qK`, `C0_qK`

这样做的直接收益是：

- graph / interface / symbolic expression 如果只需要 `H/H0`，就不会顺手把整套 `C_qK` 也 materialize 出来
- `C_qK` 只在真正访问时才构造

相关文件：

- `src/regimes.jl`


### 2. Binding regular regime 的 `H/H0` 改成基于 x-neighbor graph 的 rank-1 增量传播

加入了一个新的内部路径：

1. 先用一个 seed regular regime 直接构造 `H, H0`
2. 沿着 regime x-neighbor graph 的边，用 rank-1 更新公式传播到同一连通分量内的其他 regular regimes

对单条边，如果第 `i` 行 dominant choice 从 `j_from` 变到 `j_to`，则

```math
M' = M + e_i (e_{j_to} - e_{j_from})^\top,
\qquad
M_0' = M_0 + \delta_0 e_i,
\qquad
\delta_0 = \log\frac{L_{i,j_to}}{L_{i,j_from}}.
```

若当前 `H = M^{-1}`，`H0 = -H M0`，则

```math
c = H_{:i},
\qquad
s^\top = H_{j_to,:} - H_{j_from,:},
\qquad
\delta = 1 + H_{j_to,i} - H_{j_from,i}.
```

当 `\delta \neq 0` 时，

```math
H' = H - \frac{c s^\top}{\delta},
```

```math
H_0' = H_0 - \frac{c\left((H_0)_{j_to}-(H_0)_{j_from}+\delta_0\right)}{\delta}.
```

代码里对应：

- `_rank1_update_H_H0` in `src/matrix_inverse.jl`
- `_ensure_regular_affine_cache!` and `_propagate_regular_component!` in `src/regimes.jl`

这一步现在已经真正用在 binding regular regime 上。


### 3. 把 rank-k 版本的公式也整理成公共内部 helper

为了后续把：

- `L -> L_ss`
- `N -> N_ss`
- 不同 `N_ss` 之间的替换

也统一看成“低秩行替换”，这轮顺手把 Woodbury 形式的 affine update 公式也固化成了 helper：

```math
M' = M + U V^\top,
\qquad
M_0' = M_0 + U \delta_0.
```

则

```math
H' = H - H U (I + V^\top H U)^{-1} V^\top H,
```

```math
H_0' = H_0 - H U (I + V^\top H U)^{-1}\left(V^\top H_0 + \delta_0\right).
```

代码里对应：

- `_lowrank_update_H_H0` in `src/matrix_inverse.jl`

说明：

- 这轮真正落地使用的是 rank-1 版本
- rank-k helper 已经放进去了，方便下一步把更多 row-replacement 流程统一进来


### 4. Nullity-1 singular regime 现在也会计算 `H0 = -H M0`

以前 nullity-1 regime 只会记录 ray/adjugate-like `H`，不保存 `H0`。

现在对：

- `BindRegime`
- `BncRegime`

的 nullity-1 情况都统一计算：

```math
H_0 := -H M_0.
```

这有两个直接好处：

- interface / shared hyperplane 分析可以统一使用 `H, H0`
- symbolic 输出和调试不再被 “singular 就完全没有 H0” 卡住

注意：

- 这里的 `H` 在 singular case 下仍然是 ray/adjugate-like 对象，不应误读成真正的 affine inverse
- 但 `H0 = -H M0` 作为同尺度下的 offset，在内部几何分析里是有用的

相关文件：

- `src/regimes.jl`
- `src/Bnc_regime.jl`


### 5. Regime graph 的 qK interface 改成共享 pool，而不是双向边各存一份

以前 `VertexEdge` 的双向边会各自存：

- `change_dir_qK`
- `intersect_qK`

这在内存和构造上都存在重复。

现在做法是：

- 给 `VertexGraph` 增加一个 `qK_interface_pool`
- 对每一条无向 regime neighbor，只计算一次 qK interface
- 正向边存 `(pool_idx, +1)`
- 反向边存 `(pool_idx, -1)`
- 真正访问 `edge.change_dir_qK` 时再 lazy materialize

这样至少消掉了最直接的“两边各一份”重复。

相关文件：

- `src/initialize.jl`
- `src/regime_graphs.jl`


### 6. Mixed regime 的 steady-state `H_ss/H0_ss` 做了按 steady-state perm 的去重

在同一个 catalysis row 内，不同 binding regimes 可能对应同一个 steady-state perm。

这轮把 row-level 初始化改成：

- 先按 steady-state perm 去重
- 对每个 unique perm 只算一次 `H_ss/H0_ss`
- 再把结果分发回该 row 下所有共享这个 perm 的 `BncRegime`

这一步对 mixed regime 层面减少了明显的重复 `H/H0` 计算。

相关文件：

- `src/Bnc_regime.jl`


### 7. 修复的相关回归

- `summary_RO_path(...; show_volume=false)` 在 `Nothing` 上求和的问题
- `qK_x_mapping.jl` 中残留的 `_LN_sparse` 字段访问

相关文件：

- `src/regime_graphs.jl`
- `src/qK_x_mapping.jl`


### 8. 接口兼容性

这轮尽量没有破坏公共 API。

保持不变的点：

- `get_regime`, `get_H`, `get_C_*`, `get_interface`, `get_regimes_graph!` 等入口仍然可用
- `vertex/vertices` 的旧别名仍保留

发生的行为扩展：

- `get_H_H0` 对 nullity-1 的 `BindRegime` / `BncRegime` 现在可返回 `(H, H0)`，不再直接报错


### 9. 暂时没有完全做完的地方

虽然 rank-k 公式已经整理出来了，但这轮还没有把下面几条全部改成统一的低秩传播：

- 跨不同 `N_ss` 的 mixed-row 传播
- `C_qK`, `C_qKk_cat`, `C_qKk_ss` 的更深层 row-pool 化
- 完全移除内部的 `vertex/vertices` 类型名

当前状态是：

- 先把最容易收益最大、又不太破坏现有结构的部分做进去
- 后续如果继续推进，可以直接在已有 helper 基础上扩展


### 10. 当前验证

运行：

```bash
julia --project test/runtests.jl
```

结果：

- binding tests 通过
- notebook workflow tests 通过
- larger RO path workflow tests 通过
- catalysis / mixed regime tests 通过

总计 129 个测试通过。
