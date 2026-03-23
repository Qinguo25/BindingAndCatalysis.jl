# Changelog

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
