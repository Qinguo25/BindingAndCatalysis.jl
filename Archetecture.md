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

定义在 [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)。

它是整个项目的主模型，包含：

- binding matrices：`N`, `L`
- 维度信息：`r`, `n`, `d`
- 符号：`x_sym`, `q_sym`, `K_sym`
- 可选 catalysis：`catalysis::Union{CatalysisData,Nothing}`
- binding regimes：`BindRegimes`
- mixed regimes：`BncRegimes`
- graph cache：`vertices_graph`
- 数值辅助缓存：`IntegrationHelper`, `_L_helper`

可以把它理解成“项目里所有功能共享的根对象”。


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


### 3.6 `Regimes` 和 `VertexGraph`

`Regimes` 是一个轻量容器：

- `vertices_perm_dict`
- `vertices_data`

虽然内部字段仍沿用 `vertices_*` 的旧名字，但公共 API 已经统一到 `regime/regimes` 口径，旧名字保留在 [src/old_api.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/old_api.jl) 里作为兼容别名。

`VertexGraph` 是 regime 邻接图缓存，用于：

- 邻接关系
- interface/change direction
- SISO path 枚举


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
- 构造数值缓存和 matrix helper


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
```

这一步会：

- 从 `L` 的每一行 possible dominant choice 枚举 `perm`
- 计算 asymptotic / singular 信息
- 延迟或按需补全 `P, H, C_qK` 等缓存


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

### 6.1 基础类型与装配

- [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)
  负责定义 `Bnc`, `BindRegime`, `CatalysisData`, `CatalysisRegime`, `BncRegime`, `SISOPaths` 等核心类型，以及构造器、缓存初始化、`include` 顺序。


### 6.2 组合学枚举与矩阵辅助

- [src/find_matrix_vertex.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/find_matrix_vertex.jl)
  负责从一个矩阵的每一行 dominant choice 出发构造 regime/vertex 组合。

- [src/matrix_inverse.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/matrix_inverse.jl)
  负责 `M` 或其相关子矩阵的逆、rank-1 singular 情况下的 ray-like 信息，以及对应 cache。

- [src/helperfunctions.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/helperfunctions.jl)
  杂项矩阵/符号/索引辅助函数。


### 6.3 Binding 层

- [src/regimes.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regimes.jl)
  binding regime 的核心逻辑：初始化 `BindRegime`、计算 `P/M/H/C` 等对象、提供访问 API。

- [src/regime_assign.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regime_assign.jl)
  给定 `x` 或 `qK`，判断当前点属于哪个 regime。

- [src/qK_x_mapping.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/qK_x_mapping.jl)
  数值映射 `x ↔ qK`，包括 homotopy / nonlinear solve / trajectory。

- [src/numeric.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/numeric.jl)
  数值导数与 reaction-order 类计算。

- [src/volume_calc.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/volume_calc.jl)
  regime/polyhedron 的 Monte Carlo 体积估计。


### 6.4 Catalysis 与 mixed regime 层

- [src/Catalysis_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Catalysis_regime.jl)
  catalysis regime 的枚举、getter、条件矩阵和查询 API。

- [src/Bnc_regime.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/Bnc_regime.jl)
  mixed regime 构造、pair-based retrieval、consistency 条件、steady-state reduced map、`H_bd` 和稳定性接口。

- [src/d_stable.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/d_stable.jl)
  diagonal stability / Hurwitz 性质的数值判断。


### 6.5 图、路径、符号输出和可视化

- [src/regime_graphs.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regime_graphs.jl)
  regime neighbor graph、SISO path、路径 polyhedron、RO path 汇总。

- [src/symbolics.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/symbolics.jl)
  把内部矩阵渲染成易读表达式，是 notebook 和 debug 最常用的“解释层”。

- [src/visualize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/visualize.jl)
  图结构、路径、切片、多 regime 轨迹可视化。


### 6.6 兼容层

- [src/old_api.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/old_api.jl)
  保留旧的 `vertex/vertices` 风格 API 别名，方便老 notebook 继续运行。


## 7. 推荐阅读顺序

如果你准备维护这个项目，推荐按下面顺序读。

1. [README.md](/home/joker/Realizibility_index/BindingAndCatalysis.jl/README.md)
2. [Examples/Minimal_example.ipynb](/home/joker/Realizibility_index/BindingAndCatalysis.jl/Examples/Minimal_example.ipynb)
3. [src/initialize.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/initialize.jl)
4. [src/regimes.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regimes.jl)
5. [src/qK_x_mapping.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/qK_x_mapping.jl)
6. [src/regime_graphs.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/src/regime_graphs.jl)
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
summary_RO_path(pths; observe_x=:E)
get_volume(model, 1)
```


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

所以有些函数在 singular 分支没有显式 `H0`，这是设计使然，不是缺字段。


### 10.5 这个项目很依赖“延迟计算 + cache”

很多对象初建时并不会把所有字段都算完，而是：

- 先保存最小身份信息
- 后续按需补全

因此调试时如果看到某些字段是 `nothing`，先确认是不是还没触发对应初始化流程。


## 11. 对开发者最有用的测试与示例

- [test/runtests.jl](/home/joker/Realizibility_index/BindingAndCatalysis.jl/test/runtests.jl)
  现在是最可靠的程序化回归入口，覆盖 binding、catalysis、mixed regime，以及 notebook 的主流程。

- [Examples/Minimal_example.ipynb](/home/joker/Realizibility_index/BindingAndCatalysis.jl/Examples/Minimal_example.ipynb)
  最适合交互式学习。

- [test/work_summary_and_suggestions.md](/home/joker/Realizibility_index/BindingAndCatalysis.jl/test/work_summary_and_suggestions.md)
  记录了最近一轮关于 `CatalysisRegime` / `BncRegime` 的补全与一些设计建议。


## 12. 如果我要改功能，先看哪里

- 想改 binding regime 数学对象：看 `src/regimes.jl`
- 想改 catalysis regime：看 `src/Catalysis_regime.jl`
- 想改 mixed consistency / steady-state reduction：看 `src/Bnc_regime.jl`
- 想改 `x ↔ qK` 数值求解：看 `src/qK_x_mapping.jl`
- 想改 graph/path 分析：看 `src/regime_graphs.jl`
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
