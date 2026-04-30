# BindingAndCatalysis.jl Archetecture

这份文档面向第一次进入这个仓库的人，目标是回答 4 个问题：

1. 这个 package 在数学上到底在做什么？
2. 代码里最重要的数据类型是什么？
3. 一条完整的分析流程是怎样从输入走到输出的？
4. 如果我要改某一类功能，应该先看哪个文件？

项目名虽然叫 `BindingAndCatalysis.jl`，但它实际上包含 3 层对象：

- binding network：只看快变量平衡与守恒关系
- catalysis network：只看催化通量 regime
- mixed regime：把 binding regime 和 catalysis regime 组合起来


## 1. 一句话理解项目

这个项目把一个带时间尺度分离的生化网络拆成：

- 快层：binding equilibrium，决定 `x` 与 `(q, K)` 的关系
- 慢层：catalytic flux balance，决定哪些催化通量组合能在 steady state 自洽

然后它做 3 件事：

- 枚举 dominance regime
- 在不同坐标基底之间搬运这些 regime 的条件
- 在 regime 层面分析可行性、路径、体积、以及局部稳定性


## 2. 数学背景：最少需要知道什么

### 2.1 Binding 层

设：

- `x ∈ R^n_{>0}`：free species concentration
- `q ∈ R^d_{>0}`：total concentrations
- `K ∈ R^r_{>0}`：binding equilibrium constants

binding manifold 写成：

```math
q = Lx,
\qquad
N \log x = \log K,
\qquad
d + r = n.
```

其中：

- `L` 负责把 free species 累加成 totals
- `N` 负责表达平衡常数约束

在一个固定 binding regime 内，每个 `q_i` 都由某个 dominant monomial 近似，因此有

```math
\log q = P \log x + P_0.
```

再把 `q` 和 `K` 拼在一起，得到

```math
\log(q, K) = M \log x + M_0,
\qquad
M := \begin{bmatrix} P \\ N \end{bmatrix},
\qquad
M_0 := \begin{bmatrix} P_0 \\ 0 \end{bmatrix}.
```

如果 `M` 可逆，则

```math
\log x = H \log(q, K) + H_0.
```

这正是代码里 `BindRegime.H` / `BindRegime.H0` 的含义。


### 2.2 Catalysis 层

催化部分写成

```math
\dot q_{\mathrm{cat,dep}} = \Gamma v,
\qquad
\log v = \Pi \log x + \log k.
```

这里：

- `Γ`：催化反应对 `(q_cat, q_dep)` 的变化
- `Π`：通量指数矩阵
- `k`：催化速率常数

代码会先把坐标从 `(q_cat, q_dep)` 改写成 `(q_cat, w)`，其中 `w` 是 `Γ` 左零空间对应的守恒量。于是 reduced dynamics 变成：

```math
\dot q_{\mathrm{cat}} = S v,
\qquad
\dot w = 0,
\qquad
\dot q_{\mathrm{para}} = 0.
```

这里有一个很重要的细节：

- `S^+ v` 和 `S^- v` 一般不是单个 flux selector
- 它们是正线性映射，因此 regime 近似会带常数截距

所以对一个固定 catalysis regime，代码记录的是：

- `P_pos_neg`, `P0_pos_neg`：`f([S^+; S^-])` 的 dominant monomial map
- `Pθ = P^+ - P^-`
- `P0θ = P0^+ - P0^-`
- `Cθ = [C^+; C^-]`
- `C0θ = [C0^+; C0^-]`
- `PΠ = Pθ Π`
- `CΠ = Cθ Π`

它们分别用于：

- steady-state equation
- catalytic dominance condition

具体地：

```math
P^\theta \Pi \log x + P^\theta \log k + P_0^\theta = 0,
```

```math
C^\theta \Pi \log x + C^\theta \log k + C_0^\theta \ge 0.
```


### 2.3 Mixed regime：binding + catalysis

一个 `BncRegime` 就是：

- 1 个 `BindRegime`
- 1 个 `CatalysisRegime`

组合后，主要看 3 个坐标基底：

- `(x, k)`：最贴近原始 dominance 选择
- `(q, K, k)`：mixed consistency 的自然坐标
- `(q_ss, K, k)`：steady-state reduction 后的自然坐标，其中 `q_ss = (w, q_para)`

其中 steady-state reduction 实际上是先在 `(q_ss, K_ss)` 基底求

```math
\log x = H_{ss} \log(q_{ss}, K_{ss}) + H_{0,ss},
\qquad
\log K_{ss} = \begin{bmatrix}\log K \\ -(P^\theta \log k + P_0^\theta)\end{bmatrix},
```

再把它展开回 `(q_ss, K, k)`。

这 3 个基底非常重要。读代码时如果搞混，大多数函数都会显得“名字相似但不知区别”。


## 3. 核心对象和它们的关系

### 3.1 `Bnc`

主类型定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)；和 graph/path 相关的高层 wrapper `SISOPaths` 则已经收敛到 [src/SISO.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/SISO.jl)。

它是整个项目的主模型，包含：

- binding matrices：`N`, `L`
- 维度信息：`r`, `n`, `d`
- 符号：`x_sym`, `q_sym`, `K_sym`
- 可选 catalysis：`catalysis::Union{CatalysisData,Nothing}`
- binding regimes：`BindRegimes`
- mixed regimes：`BncRegimes`
- graph cache：`vertices_graph`
- 组合学辅助缓存：`_L_helper`
- binding affine coefficient mode：`affine_coeff_mode`
- affine / numerical cache：
  - `_regimes_affine_ready`, `_regimes_affine_lock`
  - `IntegrationHelper`, `_integration_helper_lock`

可以把它理解成“项目里所有功能共享的根对象”。

这里要特别注意两个 helper 的分工：

- `_L_helper` 是 eager 的，构造 `Bnc` 时就建立；它服务于 regime 枚举、canonical x-space hyperplane、以及邻接图构造。
- `IntegrationHelper` 是 lazy 的；现在不会在 `Bnc` 构造时立刻生成，而是在第一次数值积分 / 数值求解真正需要时，由 `_integration_helper!` 线程安全地创建并缓存。它保存：
  - homotopy 默认 anchor
  - `_LN_sparse = Float64.(sparse([L; N]))`
  - `_LN_lu`
  - top / bottom block 的稀疏索引辅助量

`Bnc` 还记录 binding 层 affine 系数的存储模式：

- `affine_coeff_mode = :float`：默认模式，`BindRegime.H` / `C_qK` 存 `Float64`
- `affine_coeff_mode = :rational`：exact mode，`BindRegime.H` / `C_qK` 存 `Rational{Int}`

这个 mode 只影响 binding 层的线性系数矩阵：

- exact：`H`, `C_qK`
- 仍保持 `Float64`：`P0`, `M0`, `H0`, `C0_x`, `C0_qK`


### 3.2 `BindRegime`

也是定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)。

它描述一个 binding dominance regime，核心字段是：

- `perm`：每一行 `q_i` 选择哪个 dominant species
- `P`, `P0`
- `M`, `M0`
- `C_x`, `C0_x`
- `nullity`
- `H`, `H0`
- `C_qK`, `C0_qK`
- `volume`

直观上：

- `perm` 是 regime 的离散标签
- `P/P0/M/M0/H/H0` 是这个 regime 下的线性化映射
- `C_*` 是这个 regime 的 admissibility 条件

现在还要额外记住：

- `H` 和 `C_qK` 可以是 `Float64` 稀疏矩阵，也可以是 `Rational{Int}` 稀疏矩阵
- `H0`、`C0_qK` 仍然是 `Float64`
- `nullity > 1` 时不会定义 `H/H0`


### 3.3 `CatalysisData`

定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)。

它挂在 `Bnc.catalysis` 下面，负责保存催化网络本身：

- `Γ`, `Π`
- `S`, `L_Γ`
- `r_v`, `n_v`, `d_w`, `d_para`
- `k_sym`
- `S_pos_neg`
- `CatalysisRegimes`

一个很重要的实现细节是：

- 在构造 `CatalysisData` 时，binding network 会被重排，使 catalytic active totals 排在前面
- 因此 `q` 的顺序默认是 `(q_cat, w, q_para)`，很多后续 API 都依赖这个约定


### 3.4 `CatalysisRegime`

定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)，主要 API 在 [src/Catalysis_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Catalysis_regime.jl)。

它对应一个催化通量 regime，核心字段：

- `perm`
- `P_pos_neg`
- `P0_pos_neg`
- `P`
- `P0`
- `C`
- `C0`
- `PΠ`
- `CΠ`

直观上：

- `P/P0` 对应 steady-state balance `P^θ log v + P0^θ = 0`
- `C/C0` 对应 dominance inequalities `C^θ log v + C0^θ \gg 0`
- `PΠ/CΠ` 是把它们搬到 `x` 基底后的系数矩阵


### 3.5 `BncRegime`

定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)，主要逻辑在 [src/Bnc_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Bnc_regime.jl)。

它是项目里最重要也最容易混淆的对象。一个 `BncRegime` 同时保存：

- 绑定的 `bind_rgm`
- 绑定的 `catalysis_rgm`
- 稳定性筛选矩阵 `H_bd`
- steady-state reduced 映射 `H`, `H0`
- `(q, K, k)` 基底下的 consistency 条件
- `(q_ss, K, k)` 基底下的 consistency 条件

要特别区分两个 `H`：

- `bind_rgm.H`：`(q, K) -> x`
- `bnc_rgm.H`：`(q_ss, K, k) -> x`

以及：

- `H_bd` 不是映射，而是 stability screening matrix

当前 mixed 层还有一个很重要的实现边界：

- binding 层若使用 `H_mode = :rational`，`bind_rgm.H` / `bind_rgm.C_qK` 可以是 exact 的
- 一旦进入 `BncRegime` 组装、mixed consistency、stability screening 这类数值流程，会显式转回 `Float64`

因此 exact mode 目前是“binding-layer exact”，不是“整个 mixed pipeline 全 exact”。


### 3.6 `Regimes`、`VertexGraph`、`SISODAG`、`SISOProblem`、`SISOHelper` 和 `SISOPaths`

`Regimes` 是一个轻量容器：

- `vertices_perm_dict`
- `vertices_data`

虽然内部字段仍沿用 `vertices_*` 的旧名字，但公共 API 已经统一到 `regime/regimes` 口径，旧名字保留在 [src/old_api.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/old_api.jl) 里作为兼容别名。

`VertexGraph` 是 regime 邻接图缓存，用于：

- 邻接关系
- interface/change direction
- SISO path 枚举

它现在不是“只有图结构”的轻量对象，而是 binding regime 图算法的核心缓存层。当前实现里：

- `neighbors[u]` 是 `VertexEdge` 列表
- 每条边都记录：
  - 改变的是哪一行 `i`
  - 对应的 x-space hyperplane 在全局池中的索引 `c_c0_x_idx`
  - 该边方向下的符号 `c_c0_x_sign`
  - 对应的 qK-space hyperplane 在全局池中的索引 `qK_interface_idx`
  - 该边方向下的符号 `qK_interface_sign`
- `x_interface_pool` 直接复用 `MatrixHelper.hyperplanes`
- `qK_interface_pool` 是专门给 qK-space 接口做的全局去重池

这意味着 qK-space 超平面现在和 x-space 一样，是“总体只存一份”的设计：

- edge 不再各自持有完整的 qK interface 向量
- 正反两条边共享同一个 `qK_interface_idx`
- 方向差异只靠 `qK_interface_sign = ±1` 表示

这套设计把 graph cache 从“邻接表”提升成了“邻接表 + canonical interface pool”。

要注意一个最近的结构变化：

- `VertexGraph` 现在不再长期保存单独的 `x_grh` 字段
- x-neighbor graph 由 `neighbors` 按需通过 `get_neighbor_graph_x(...)` 重建
- 因此 `VertexGraph` 更像“最小必要邻接缓存”，而不是把每种图表示都各存一份

在当前架构里，`VertexGraph` 负责“全局 regime 图缓存”，而 SISO 路径条件求解已经集中到 [src/SISO.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/SISO.jl)。这一层又被拆成 4 个层次：

- `SISODAG`
  - 保存某个 `change_qK` 方向下的有向图
  - 保存 `sources` / `sinks`
  - 保存 reachability bitmatrix
  - 它表达的是“图论问题本身”

- `SISOProblem`
  - 保存 `bn`
  - 保存 `change_qK_idx`
  - 保存 `dag::SISODAG`
  - 它表达的是“某个模型、某个坐标轴方向下的一次 SISO 分析问题”

- `SISOHelper`
  - 是内部 memoized backend
  - 只保存求解缓存，不再长期保存对外结果
  - 当前主要缓存：
    - `vertex_prisms`
    - `interface_prisms`
    - `pair_conditions`
  - `pair_conditions[(from,to)]` 对应一个 `Dict(path_tuple => polyhedron)`，因此缓存粒度已经从“矩阵里挂 path 对象”收口成“按 source-sink pair 缓存条件映射”

- `SISOPaths`
  - 是对外的路径分析对象
  - 现在只保存：
    - `problem`
    - `rgm_paths`
    - lazy 的 `path_index`
    - lazy 的 `condition_helper`
    - `path_polys` / `path_volume` 及其状态位
  - 它表达的是“某一个坐标轴方向上的全路径结果容器”

这次重构的核心目标是把“图问题定义”、“递归求解缓存”、“对外路径结果”三层拆开，减少 `RegimePath`/helper/path container 之间的重复存储。调用 `get_polyhedra` / `get_volume` / `get_RO_paths` 时，底层都会复用同一个 `SISOHelper`，但外层不再把 helper 内部缓存结构直接暴露成长期数据模型。


### 3.7 `MatrixHelper` 和 `IntegrationHelper`

这两个 helper 现在分别服务于“组合学层”和“数值层”。

`MatrixHelper`：

- 来自 `L`
- 保存每行可选 dominant monomial、constraint row partition、choice map
- 建立 canonical 的 x-space hyperplane pool
- 被 regime 枚举、graph 构造、rank-1 propagation 直接复用

`IntegrationHelper`：

- 来自 `(L, N)`
- 保存 homotopy / nonlinear solve 常用的 `_LN_sparse`、`_LN_lu`
- 保存 top block / bottom block 稀疏索引
- 默认 anchor 也放在这里
- 现在采用 lazy + lock 的方式缓存，避免构造 `Bnc` 时做不必要的数值预处理，也避免多线程首次调用时重复计算

因此常用的 `Float64.(sparse([L; N]))` 现在不会在每次积分 / mapping 时重建，而是复用 helper 内的缓存；需要可变副本时再 `copy(...)` 或 `deepcopy(...)`。


## 4. 代码的整体流水线

最核心的流水线可以分成 7 步。

### 4.1 构造 binding 网络

入口：

```julia
model = Bnc(; N=..., L=..., x_sym=..., q_sym=..., K_sym=...)
```

这一步会：

- 验证 `N, L` 维度
- 自动生成缺失的 `L` 或 `N`
- 立刻构造 `_L_helper`
- 只保留数值 helper 的 lazy 入口，不会预先计算 `IntegrationHelper`


### 4.2 可选：附加 catalysis 网络

入口：

```julia
update_catalysis!(model; Γ=..., Π=..., k_sym=..., q_picked=...)
```

这一步会：

- 构造 `CatalysisData`
- 计算 `S`, `L_Γ`
- 重排 binding network 的 `q` 顺序
- 建立 catalysis regime 枚举所需辅助结构


### 4.3 枚举 binding regimes

入口：

```julia
find_all_regimes!(model)
# 或
find_all_regimes!(model; H_mode = :rational)
```

这一步会：

- 从 `L` 的每一行 possible dominant choice 枚举 `perm`
- 立刻用 `all_perms` 和 `model._L_helper` 构造 x-neighbor regime graph
- 先只建立轻量的 `BindRegime` 容器对象
- 根据 `H_mode` 决定 binding affine 系数存成 `Float64` 还是 `Rational{Int}`
- 然后进入 `_prefill_affine_cache!`：
  - 按 x-graph connected component 处理
  - 在每个 component 里挑 seed
  - seed 先直接计算 `P/P0/M/M0/C_x/C0_x`
  - 若 seed regular，则沿图用 rank-1 update 传播 `H/H0`
  - 传播过程中即时给可判定的 regime 打上 `nullity = 0/1`
  - 若 seed 是 `nullity = 1`，只直接计算自身，不再继续传播
- 只有剩下的高 nullity 候选才 defer 到 `_calc_nullity`
- 最后 `_ensure_full_regimes_graph!` 再把所有可定义的 qK-space interface 补进 `VertexGraph.qK_interface_pool`

这里有两个重要实现选择：

- `_prefill_affine_cache_core!` 不会把全部 regime 的 nullity 重新显式算一遍；它保留“图上传播 + 高 nullity defer”的原逻辑
- exact mode 下会优先找 `nullity == 0` 的 regular seed，再沿图传播回 singular regime；只有找不到 regular seed 时，才退回 exact singular fallback
- 多线程传播工作区 `AffinePropagateWorkspace` 现在按 `Threads.maxthreadid()` 分配槽位，而不是 `Threads.nthreads()`，避免 notebook / task 调度下的线程槽越界

另外，exact mode 的边界是：

- rank-1 propagation 会保持 `H` 与 `C_qK` 的 exact 性质
- `H0` / `C0_qK` 继续走 `Float64`
- `get_polyhedron(...)` 在真正交给 `Polyhedra.jl` / `CDDLib` 之前，会把 exact 系数转成 `Float64`


### 4.4 枚举 catalysis regimes

入口：

```julia
find_catalysis_regimes!(model)
```

这一步是在 `S_pos_neg` 上做类似的 dominance 枚举，得到所有 `CatalysisRegime`。


### 4.5 匹配 mixed regimes

入口：

```julia
match_regimes!(model)
```

这一步会把 binding regimes 和 catalysis regimes 逐对组合，生成 `model.BncRegimes` 这个矩阵。索引顺序是：

- 行：catalysis regime
- 列：binding regime

所以：

```julia
rgm = get_bnc_regime(model, bind_perm, cat_perm)
```

底层对应的是 `model.BncRegimes[cat_idx, bind_idx]`。


### 4.6 做一致性、路径和体积分析

常见入口：

- `get_C_C0_nullity`
- `get_polyhedron`
- `get_volume`
- `get_regimes_graph!`
- `SISOPaths(...)`
- `get_RO_path` / `summary_RO_path`

当前路径分析的实际流水线是：

1. `get_regimes_graph!(model; full=true)` 先保证 `VertexGraph` 和 qK interface direction 都已 materialize。
2. `SISOPaths(model, change_qK)` 在 [src/SISO.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/SISO.jl) 里先构造 `SISODAG`，再封装成 `SISOProblem`；这一层会沿单个坐标轴定向并过滤 singular isolated regimes。
3. 如果没有手动传 `rgm_paths`，会先枚举所有 source-to-sink 路径；这一步现在带进度提示。
4. `get_polyhedra(pths)` 会懒触发 `SISOHelper`，按 source-sink pair 批量求条件；当前缓存是 `vertex prism + interface prism + pair_conditions[(from,to)]`，并带进度提示。
5. `get_volume`、`get_expression_path`、`get_RO_paths`、`summary_RO_path` 都建立在这些 path polyhedra 之上。

所以现在的“全路径条件”实现已经不再分散在多个文件里，而是统一收束到 `src/SISO.jl`。


### 4.7 做符号渲染和稳定性判断

常见入口：

- `show_condition_*`
- `show_expression_*`
- `show_catalysis_dynamics`
- `show_reduced_catalysis_dynamics`
- `is_stable`
- `judge_stability!`


## 5. 三个最重要的坐标基底

这是新手最值得反复看的一节。

### 5.1 `x` 或 `(x, k)`

用途：

- 原始 dominance 选择
- catalytic regime 条件

典型函数：

- `show_condition_x`
- `show_condition_xk`


### 5.2 `(q, K)` 或 `(q, K, k)`

用途：

- 把 binding regime 从 `x` 空间搬到总量参数空间
- mixed regime consistency

典型函数：

- `show_condition_qK`
- `show_condition_qKk`


### 5.3 `(q_ss, K, k)`，其中 `q_ss = (w, q_para)`

用途：

- steady-state reduction
- 消去 `q_cat`
- 获得 reduced consistency 条件和 `q_cat` 显式表达
- 吸收 `P0^θ` 对 `K_ss` 的平移效应

典型函数：

- `show_condition_qssKk`
- `show_expression_qcat`


## 6. 源码地图：每个文件大概负责什么

下面按“读代码的推荐顺序”列出。

### 6.1 模块装配与基础类型

- [src/BindingAndCatalysis.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/BindingAndCatalysis.jl)
  负责模块外壳、`include` 顺序、export，以及把各子文件装配成最终 public API。

- [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)
  负责定义 `Bnc`, `BindRegime`, `CatalysisData`, `CatalysisRegime`, `BncRegime` 等核心类型、构造器，以及 lazy numerical cache 的基础脚手架。`SISOPaths` 已不在这个文件里。


### 6.2 组合学枚举与矩阵辅助

- [src/Mathcore/find_matrix_vertex.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Mathcore/find_matrix_vertex.jl)
  负责从一个矩阵的每一行 dominant choice 出发构造 regime/vertex 组合。

- [src/Mathcore/perm_graph_core.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Mathcore/perm_graph_core.jl)
  负责 x-neighbor graph 构造、component 级 affine propagation、qK interface pool 去重、以及 graph cache 的补全。

- [src/Mathcore/matrix_inverse.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Mathcore/matrix_inverse.jl)
  负责 `M` 或其相关子矩阵的逆、adjugate/nullity-1 情况下的 affine 信息，以及 rank-1 update 公式。现在同时包含 float mode、exact mode、以及 exact-aware sparse rank-1 update 的实现。

- [src/helperfunctions.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/helperfunctions.jl)
  杂项矩阵/符号/索引辅助函数。


### 6.3 Binding 层

- [src/regimes.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regimes.jl)
  binding regime 的核心逻辑：初始化 `BindRegime`、计算 `P/M/H/C` 等对象、提供访问 API。`find_all_regimes!(...; H_mode=...)`、`_materialize_qK_conditions!`、以及 exact/float mode 切换逻辑都在这里。

- [src/regime_assign.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regime_assign.jl)
  给定 `x` 或 `qK`，判断当前点属于哪个 regime。

- [src/qK_x_mapping.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/qK_x_mapping.jl)
  数值映射 `x ↔ qK`，包括 homotopy / nonlinear solve / trajectory；当前会按需拉起 `IntegrationHelper`，并复用缓存的 `_LN_sparse` / `_LN_lu`。

- [src/numeric.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/numeric.jl)
  数值导数与 reaction-order 类计算。

- [src/volume_calc.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/volume_calc.jl)
  regime/polyhedron 的 Monte Carlo 体积估计。


### 6.4 Catalysis 与 mixed regime 层

- [src/Catalysis_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Catalysis_regime.jl)
  catalysis regime 的枚举、getter、条件矩阵和查询 API。

- [src/Bnc_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Bnc_regime.jl)
  mixed regime 构造、pair-based retrieval、consistency 条件、steady-state reduced map、`H_bd` 和稳定性接口。这里也负责把 binding exact 系数在 mixed 边界显式转回 `Float64`。

- [src/Mathcore/d_stable.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Mathcore/d_stable.jl)
  diagonal stability / Hurwitz 性质的数值判断。


### 6.5 图、路径、符号输出和可视化

- [src/SISO.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/SISO.jl)
  当前 graph/path backend 的主实现文件。它按层组织了：
  - `VertexGraph` 访问与 graph utility
  - axis-aligned SISO 有向图构造
  - `SISODAG` / `SISOProblem`
  - polyhedron projection / prism helper
  - 内部 `SISOHelper`
  - 对外 `SISOPaths`
  - path polyhedron、volume、expression tracing、reaction-order、summary API

- [src/regime_graphs.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regime_graphs.jl)
  当前是空的过渡文件，只保留为兼容/装配占位；实质 graph/path 逻辑已经全部迁入 `src/SISO.jl`。

- [src/symbolics.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/symbolics.jl)
  把内部矩阵渲染成易读表达式，是 notebook 和 debug 最常用的“解释层”。

- [src/visualize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/visualize.jl)
  图结构、路径、切片、多 regime 轨迹可视化。`draw_graph(model; hide_nullity_ge_2=true)` 可以在画图时直接隐藏 `nullity >= 2` 的 binding regime 节点。


### 6.6 兼容层

- [src/old_api.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/old_api.jl)
  保留旧的 `vertex/vertices` 风格 API 别名，方便老 notebook 继续运行。


## 7. 推荐阅读顺序

如果你准备维护这个项目，推荐按下面顺序读。

1. [README.md](/home/joker/Realizibility_index/BindingAndCatalysis.jl/README.md)
2. [Examples/Minimal_example.ipynb](/home/joker/Realizibility_index/BindingAndCatalysis.jl/Examples/Minimal_example.ipynb)
3. [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)
4. [src/regimes.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regimes.jl)
5. [src/SISO.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/SISO.jl)
6. [src/qK_x_mapping.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/qK_x_mapping.jl)
7. [src/Catalysis_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Catalysis_regime.jl)
8. [src/Bnc_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Bnc_regime.jl)
9. [src/symbolics.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/symbolics.jl)
10. [test/runtests.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/test/runtests.jl)

如果你只想先学会“怎么用”，前 3 步就够了。


## 8. 最常用的 API 入口

### 8.1 构造与更新模型

```julia
model = Bnc(; N=..., x_sym=..., q_sym=..., K_sym=...)
update_catalysis!(model; Γ=..., Π=..., q_picked=..., k_sym=...)
```


### 8.2 枚举 binding regime

```julia
find_all_regimes!(model)
# 或
find_all_regimes!(model; H_mode = :rational)
rgm = get_regime(model, 1)
rgms = get_regimes(model)
```


### 8.3 枚举 catalysis regime

```julia
find_catalysis_regimes!(model)
cat = get_catalysis_regime(model, 1)
```


### 8.4 组合 mixed regime

```julia
match_regimes!(model)
bnc_rgm = get_bnc_regime(model, bind_perm, cat_perm)
```


### 8.5 看条件和表达式

```julia
show_condition_x(rgm)
show_condition_qK(rgm)

show_condition_xk(cat)
show_condition_qKk(bnc_rgm)
show_condition_qssKk(bnc_rgm)
show_expression_qcat(bnc_rgm)
```


### 8.6 图、路径和体积

```julia
get_regimes_graph!(model)
pths = SISOPaths(model, :tS)
get_polyhedra(pths)
summary_RO_path(pths; observe_x=:E)
get_volume(model, 1)
```

其中：

- `SISOPaths(model, :tS)` 会沿 `:tS` 对应的 qK 坐标构造 SISO DAG
- `get_polyhedra(pths)` 会触发当前统一的 path-condition backend
- 如果路径很多，路径枚举和 path-condition 求解都会显示进度


### 8.7 稳定性

```julia
is_stable(bnc_rgm)
judge_stability!(bnc_rgm)
```


## 9. 一个最短上手流程

下面是最小的实践路径。

```julia
using BindingAndCatalysis

model = let
    N = [1 1 -1]
    x_sym = [:E, :S, :C]
    q_sym = [:tE, :tS]
    K_sym = [:K]
    Bnc(N=N, x_sym=x_sym, q_sym=q_sym, K_sym=K_sym)
end

find_all_regimes!(model)
rgm = get_regime(model, 1)

show_condition_x(rgm)
show_condition_qK(rgm)
show_expression_x(rgm)
```

如果你要继续到 catalysis：

```julia
update_catalysis!(
    model;
    Γ = [1 -1],
    Π = [1 0 0; 0 1 0],
    q_picked = [:tE],
    k_sym = [:k1, :k2],
)

find_catalysis_regimes!(model)
match_regimes!(model)

cat = get_catalysis_regime(model, 1)
bnc_rgm = get_bnc_regime(model, get_perm(rgm), get_perm(cat))
```


## 10. 新人最容易混淆的点

### 10.1 “regime” 和 “vertex” 基本是同一概念

历史上这个项目大量使用 `vertex/vertices` 术语。现在公共 API 逐步统一成 `regime/regimes`，但旧别名仍然保留。


### 10.2 `q` 的顺序在有 catalysis 时不是任意的

加入 catalysis 后，`q` 会被重排为：

```text
(q_cat, w, q_para)
```

这个顺序决定了：

- `q_cat_sym`
- `w_sym`
- `q_para_sym`
- `q_ss_sym`
- `show_expression_qcat`

等函数的含义。


### 10.3 `H_bd` 不是 `H`

- `H_bd` 用于稳定性筛选
- `H` / `H0` 是坐标映射或 reduced inverse

不要把它们混成一个对象。


### 10.4 singular regime 的处理方式不同

regular 情况下很多东西可直接通过矩阵逆得到；singular 情况下项目更偏向：

- 保留扩展变量
- 用 polyhedral elimination 直接消元

不过现在 `nullity = 1` 的 regime 也会保存同尺度下的 `H0 = -H M0`，方便做 interface 和几何分析。
真正需要延后到 `_calc_nullity` 批量补齐的，是传播中识别出的 `nullity >= 2` 候选。

而且 exact mode 下也不再默认要求 singular seed 一开始就硬算 adjugate：

- 会优先找 regular seed
- 从 regular regime 沿图传播回 singular regime
- 只有确实找不到 regular seed 时，才退回 exact singular fallback


### 10.5 这个项目很依赖“预填充 + 按需 materialize”

现在 binding regime 初始化大致分两层：

- `find_all_regimes!` 时就预填充：
  - regime graph
  - 基础矩阵字段
  - 通过 graph 传播得到的大部分 `nullity`
  - `nullity <= 1` 的 `H/H0`
- 最后只对 deferred 的高 nullity perm 批量调用 `_calc_nullity`
- 真正按需 materialize 的主要是：
  - `C_qK`, `C0_qK`
  - polyhedron / volume
  - qK-space graph interface

因此调试时如果看到某些字段是 `nothing`，先区分它属于“高 nullity 无法定义”还是“尚未 materialize”。


### 10.6 数值 helper 现在是 lazy + thread-safe

以前容易误以为 `Bnc` 构造完成后，数值积分相关缓存都已经准备好了。现在不是这样：

- `_L_helper` 仍然是 eager
- `IntegrationHelper` 则是 lazy

所以如果你在调试 `qK_x_mapping.jl` 或 `assign_regime_x`，看到第一次调用会经过 `_integration_helper!`，这是当前设计的一部分，不是多余绕路。

这么做的目的有两个：

- 避免很多只做组合学/图分析的 workflow 白算一份 `_LN_sparse` 和 `_LN_lu`
- 避免多线程第一次进入数值入口时重复初始化

数值入口现在会直接复用 helper 内缓存的 `_LN_sparse` 与 `_LN_lu`。


### 10.7 qK-space graph interface 现在是 pooled，而不是 edge-local

如果你以前见过“edge 直接挂一整条 qK 超平面”的旧实现，需要更新这个 mental model：

- x-space hyperplane 和 qK-space hyperplane 现在都是 pool 化管理
- edge 只存 pool index 和 sign
- 正反边共享同一份几何对象

这让 graph cache 更紧凑，也让“同一接口的正反方向”保持严格一致。


### 10.8 exact mode 的边界要分清

现在的 exact/rational 设计是分层的：

- binding coefficient matrices：可以 exact
- log offsets：继续 `Float64`
- polyhedron / volume / mixed regime / stability：进入这些数值或外部库接口前会转成 `Float64`

所以如果你在 debug 时看到：

- `get_H(model, i)` 是 `Rational`
- `get_H0(model, i)` 是 `Float64`
- mixed `BncRegime.H` 又回到了 `Float64`

这不是不一致，而是当前架构有意画出的边界。


## 11. 对开发者最有用的测试与示例

- [test/runtests.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/test/runtests.jl)
  现在是最可靠的程序化回归入口，覆盖 binding、catalysis、mixed regime，以及 notebook 的主流程。

- [Examples/Minimal_example.ipynb](/home/joker/Realizibility_index/BindingAndCatalysis.jl/Examples/Minimal_example.ipynb)
  最适合交互式学习。

### 11.1 测试目录组织约定

`test/runtests.jl` 应该保持为 package-level correctness suite：也就是 CI 和普通开发时最先运行的入口。

当某个模块需要更多专门的测试、诊断脚本、benchmark、长时间运行脚本或 reference note 时，请在 `test/` 下创建模块子目录，而不是继续把所有文件堆在 `test/` 根目录。例如：

- SISO / path-condition 相关脚本放在 [test/SISO_test/](test/SISO_test/)，并按用途分层，例如 `benchmarks/`、`diagnostics/`、`long_runs/`、`references/`、`docs/`
- 未来如果有 catalysis 专门诊断，可以放在 `test/Catalysis_test/`
- 未来如果有 visualization 专门 smoke test，可以放在 `test/Visualize_test/`

推荐区分三类文件：

- `test/runtests.jl`：快速、确定性的主回归测试，适合 CI。
- `test/<Module>_test/benchmarks/*.jl`：模块相关 benchmark。生成结果应写入忽略的 `results/` 子目录。
- `test/<Module>_test/diagnostics/*.jl`：探索性诊断脚本。
- `test/<Module>_test/long_runs/*`：远程、过夜或手动启动的长时间运行脚本。
- `test/<Module>_test/references/*.md` 和 `docs/*.md`：可复现 benchmark 的 reference note、设计说明或工作记录。

不要提交本地运行生成的 status/result JSON、stdout/stderr log、launcher log、session handoff note 等文件；这些应该通过 `.gitignore` 忽略。需要保留性能结论时，用小的、手写的 reference `.md` 总结，而不是提交整份机器输出。

SISO 的 DAG path-condition solver 在多线程下默认使用 pair+chunk queue scheduler：已满足依赖的 pair 进入全局队列，大的 middle-join pair 会进一步拆成按估计 entry 数平衡的 chunk task。单线程或 `BNC_SISO_DAG_SCHEDULER=serial` 时使用串行 DAG 调度；`BNC_SISO_DAG_SCHEDULER=queue` 可显式请求多线程队列调度。chunk 大小由自适应估计器控制，主参数是 `BNC_SISO_DAG_TARGET_CHUNK_SECONDS`，默认 `40` 秒；size / width / thread 三个 gate 默认开启，并可分别用 `BNC_SISO_DAG_CHUNK_SIZE_GATE`、`BNC_SISO_DAG_CHUNK_WIDTH_GATE`、`BNC_SISO_DAG_CHUNK_THREAD_GATE` 关闭。benchmark 输出会包含 chunk 数、chunk 估计 entry、chunk runtime、finalize time、gate skip 计数和估计 entries/sec 等字段，用于后续自动阈值调优。


## 12. 如果我要改功能，先看哪里

- 想改 binding regime 数学对象：看 `src/regimes.jl`
- 想改 catalysis regime：看 `src/Catalysis_regime.jl`
- 想改 mixed consistency / steady-state reduction：看 `src/Bnc_regime.jl`
- 想改 `x ↔ qK` 数值求解：看 `src/qK_x_mapping.jl`
- 想改 graph/path 分析：看 `src/SISO.jl`
- 想改显示输出：看 `src/symbolics.jl`
- 想改画图：看 `src/visualize.jl`
- 想保留旧 notebook 兼容性：看 `src/old_api.jl`


## 13. 总结

这个项目最核心的思想可以压缩成一句话：

> 先把 binding 和 catalysis 各自做成 regime-level 的离散描述，再把它们在合适的坐标基底里拼起来，最后在 regime 层面做可行性、路径和稳定性分析。

如果你能先抓住下面这 4 个对象，这个仓库基本就不会迷路：

- `Bnc`
- `BindRegime`
- `CatalysisRegime`
- `BncRegime`

再配合下面这 3 个基底：

- `(x, k)`
- `(q, K, k)`
- `(q_ss, K, k)`

基本就能把大多数函数名读懂。
