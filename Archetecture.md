# BindingAndCatalysis.jl Architecture

这份文档面向维护者，目标是用尽可能少的层次解释 4 件事：

1. 这个包在数学上做什么。
2. 运行时的核心对象是什么。
3. 一条典型 workflow 如何从输入走到输出。
4. 改某类功能时应该先看哪里。

如果需要看 `src/` 内部 helper / 数据结构盘点，以及哪些逻辑已经开始重复、适合继续抽象，补充材料见 [Source_helpers_and_structs.md](./Source_helpers_and_structs.md)。


## 1. 项目一句话

这个包把带时间尺度分离的生化系统拆成两层：

- fast layer: binding equilibrium
- slow layer: catalysis / steady-state consistency

然后在 regime level 做：

- dominance regime 枚举
- 条件从 `x`、`(q,K)`、`(w,K,k)` 等坐标之间搬运
- 邻接图、路径、体积、稳定性分析


## 2. 数学心智模型

### 2.1 Binding layer

binding 侧的核心关系是

```math
q = Lx,
\qquad
N \log x = \log K.
```

在一个固定 binding regime 内，每个 `q_i` 选定 dominant monomial，于是得到

```math
\log q = P \log x + P_0.
```

把 `q` 和 `K` 合在一起，得到

```math
\log(q,K) = M \log x + M_0,
\qquad
M = \begin{bmatrix} P \\ N \end{bmatrix}.
```

若 `M` 可逆，则

```math
\log x = H \log(q,K) + H_0.
```

因此 binding regime 的核心是两类对象：

- 映射：`P/P0`, `M/M0`, `H/H0`
- 条件：`C_x/C0_x`, `C_qK/C0_qK`


### 2.2 Catalysis layer

催化层写成

```math
\dot q_{\mathrm{cat,dep}} = \Gamma v,
\qquad
\log v = \Pi \log x + \log k.
```

经过坐标重排后，代码实际上在处理

- steady-state equalities
- catalytic dominance inequalities

并把它们保存为：

- `P/P0`
- `C/C0`
- `PΠ`
- `CΠ`

这里要记住：catalysis regime 本身并不直接决定 `x`，它只给出“给定 `x,k` 时哪些 flux dominance / consistency 成立”。


### 2.3 Mixed regime

一个 `BncRegime` 是：

- 一个 `BindRegime`
- 一个 `CatalysisRegime`

然后把两边的条件搬运到两个最常用坐标：

- `(q,K,k)`
- `(w,K,k)`，其中 `w = [w_old; q_para]`

这一步本质上是：

- 组合 binding 映射
- 加入 catalysis steady-state consistency
- 必要时通过 polyhedral elimination 消去中间变量


## 3. 三个必须分清的坐标系

### 3.1 `x` 或 `(x,k)`

用途：

- 原始 dominance 条件
- 催化通量条件

常见 API：

- `show_condition_x`
- `show_condition_xk`


### 3.2 `(q,K)` 或 `(q,K,k)`

用途：

- 把 binding regime 从 `x` 空间搬到总量参数空间
- mixed consistency 的主坐标

常见 API：

- `show_condition_qK`
- `show_condition_qKk`


### 3.3 `(w,K,k)`

用途：

- steady-state reduction
- 消去 `q_cat`
- 得到 reduced consistency

常见 API：

- `show_condition_wKk`
- `show_expression_qcat`


## 4. 运行时核心对象

### 4.1 `Bnc`

定义入口在 `src/initialize.jl`。

它是整个包的根对象，持有：

- binding matrices: `N`, `L`
- symbols: `x_sym`, `q_sym`, `K_sym`
- optional catalysis data
- regime caches
- regime graph cache
- helper caches
- affine mode, 即 `:float` 或 `:exact`

可以把它理解成“模型 + 所有惰性缓存”。


### 4.2 `BindRegime`

表示一个 binding dominance regime，核心字段是：

- `perm`
- `P/P0`
- `M/M0`
- `H/H0`
- `C_x/C0_x`
- `C_qK/C0_qK`
- `nullity`

`perm` 是离散标签，其余是该 regime 的线性/仿射描述。


### 4.3 `CatalysisData` 与 `CatalysisRegime`

`CatalysisData` 保存催化网络及其辅助分解。

`CatalysisRegime` 保存一个 flux dominance regime 的：

- equalities
- inequalities
- 从 flux 坐标搬到 `x` 的矩阵


### 4.4 `BncRegime`

这是 mixed 层的核心对象。它保存：

- `bind_rgm`
- `catalysis_rgm`
- `(q,K,k)` consistency
- `(w,K,k)` consistency
- steady-state reduced mapping
- `H_bd` 和稳定性相关对象

要特别区分：

- `bind_rgm.H`：`(q,K) -> x`
- `bnc_rgm.H`：`(w,K,k) -> x`
- `H_bd`：稳定性筛选矩阵，不是坐标映射


### 4.5 `Regimes` 与 `RegimeGraph`

`Regimes` 是 regime 容器。

`RegimeGraph` 是 binding regime graph cache。它不只是图结构，还包含：

- 邻接关系
- edge metadata
- x-space shared hyperplane pool
- qK-space shared hyperplane pool

这层是后续 `SIMOPaths`、regime assignment、体积估计的基础。


### 4.6 `SIMOPaths`

`SIMOPaths` 是“固定一条 q/K 方向，跟踪这条单参数变化下所有 `x` 输出”的工作对象。

它缓存：

- SIMO graph
- source / sink
- all regime paths
- node / edge / path polyhedra
- path volumes

这是路径条件和 RO path 分析的入口。


## 5. 典型 workflow

### 5.1 Binding-only workflow

```julia
model = Bnc(...)
find_all_regimes!(model)
rgm = get_regime(model, 1)
show_condition_x(rgm)
show_condition_qK(rgm)
```

内部顺序是：

1. 构造 `Bnc`
2. 枚举 dominance choices
3. 建立 x-neighbor graph
4. 预填充 affine cache
5. 对高 nullity 情况延迟补齐
6. 按需 materialize `C_qK`, polyhedron, volume


### 5.2 Catalysis / mixed workflow

```julia
update_catalysis!(model; ...)
find_catalysis_regimes!(model)
match_regimes!(model)
bnc_rgm = get_bnc_regime(model, bind_perm, cat_perm)
show_condition_qKk(bnc_rgm)
show_condition_wKk(bnc_rgm)
```

内部顺序是：

1. 构造 `CatalysisData`
2. 枚举 `CatalysisRegime`
3. 将 binding / catalysis regime 做笛卡尔匹配
4. 在 mixed 层构造一致性条件和 reduced map


### 5.3 Graph / path / volume workflow

```julia
grh = get_regimes_graph!(model; full=true)
simo = SIMOPaths(model, 1)
polys = get_polyhedra(simo)
vols = get_volumes(simo)
```

内部顺序是：

1. 构造或补全 `RegimeGraph`
2. 根据 graph 枚举路径
3. 为路径计算 node / edge / path polyhedra
4. 用 Monte Carlo 估体积


## 6. 多面体后端架构

当前多面体层是两层结构。

### 6.1 `ExactTypes`

文件：`src/ExactTypes.jl`

负责 `ExactLogExpr`。这是项目共享的 exact-log 标量层，不属于某个特定多面体后端。


### 6.2 `Polyhedra.jl` + `CDDLib.jl`

相关文件：

- `src/BindingAndCatalysis.jl`
- `src/utils/poly_backend_utils.jl`
- `src/utils/poly_utils.jl`

职责：

- 统一使用 `Polyhedra.Polyhedron` 作为多面体对象
- 用 `CDDLib.Library(:float)` 承担 H-rep 消元、相交、canonicalization 等后端计算
- 在项目内维护 `(C, C0, nullity)` 与 `Polyhedron` 之间的转换

约定上，项目内部矩阵语义统一为：

```math
C x + C_0 \ge 0
```

`Polyhedra` / `CDDLib` 的 materialize 路径只在真正需要多面体后端时触发，例如：

- singular regime 的 `qK`-space H-rep 投影
- `get_polyhedron`
- `SIMO` 路径条件的 polyhedron 运算

当前策略是：

- graph propagation 和 `H/H0` 计算固定保持 exact
- invertible regime 的 `C_qK/C_{0,qK}` 也固定保持 exact
- 一旦进入多面体后端，就把数据转成 `Float64` 并交给 `CDDLib`


## 7. 源码地图

### 7.1 根入口

- `src/BindingAndCatalysis.jl`

职责：

- include 顺序
- exports
- 顶层类型和公共 glue code


### 7.2 Binding 核心

- `src/regimes.jl`
- `src/regime_assign.jl`
- `src/qK_x_mapping.jl`
- `src/numeric.jl`
- `src/volume_calc.jl`


### 7.3 Catalysis 与 mixed

- `src/Catalysis_regime.jl`
- `src/Bnc_regime.jl`
- `src/mixed_regime/`

`src/Bnc_regime.jl` 现在只是 mixed 层入口壳，实际实现按职责拆在 `src/mixed_regime/`：

- `bnc_core.jl`
- `bnc_conditions.jl`
- `bnc_initialization.jl`
- `bnc_display.jl`


### 7.4 图和路径

- `src/regime_graphs.jl`
- `src/SIMO.jl`
- `src/simo/`
- `src/Mathcore/perm_graph_core.jl`
- `src/Mathcore/graph_propagate.jl`

`src/SIMO.jl` 现在只是入口壳，具体拆在：

- `src/simo/core.jl`
- `src/simo/polyhedra.jl`
- `src/simo/reaction_order.jl`
- `src/simo/display.jl`


### 7.5 数学核心与辅助

- `src/Mathcore/find_matrix_vertex.jl`
- `src/Mathcore/matrix_inverse.jl`
- `src/Mathcore/d_stable.jl`
- `src/helperfunctions.jl`
- `src/utils/`

`src/helperfunctions.jl` 现在只是 utilities 入口壳，具体按用途拆在 `src/utils/`：

- `matrix_utils.jl`
- `model_utils.jl`
- `symbolic_utils.jl`
- `graph_utils.jl`
- `poly_utils.jl`
- `misc_utils.jl`


### 7.6 输出层

- `src/symbolics.jl`
- `src/output/`
- `src/visualize.jl`
- `src/visualization/`
- `src/old_api.jl`

`src/symbolics.jl` 现在只保留公共 API 入口，内部拆在 `src/output/`：

- `symbolic_symbols.jl`
- `symbolic_renderers.jl`
- `symbolic_api.jl`
- `symbolic_paths.jl`

`src/visualize.jl` 同样只是入口壳，具体拆在 `src/visualization/`：

- `graphs.jl`
- `simo_plot.jl`
- `rop.jl`
- `poly_slices.jl`


## 8. 当前重要设计边界

### 8.1 exact / float 边界

当前没有 `find_all_regimes!(...; mode=...)` 这个分支。

现在应理解为：

- `find_all_regimes!` 固定用 exact 逻辑构造 affine 数据
- `H/H0`、invertible regime 的 `C_qK/C_{0,qK}` 保持 exact
- 只有多面体后端调用会显式转成 `Float64`

因此“exact”是 affine 层的默认行为，而不是要求整个仓库在所有路径上都保持 exact 标量。


### 8.2 graph cache 是 pooled hyperplane 设计

qK-space interface 不再是 edge-local 完整向量，而是：

- graph 内部维护 pool
- edge 只存 `idx + sign`

这样 regime assignment、interface 查询、volume sampling 才能共享几何对象。


### 8.3 大量对象是 lazy materialize

不是所有字段在 `find_all_regimes!` 时都立刻算完。

典型 lazy 对象包括：

- `C_qK/C0_qK`
- polyhedron
- volume
- 部分图接口和路径对象

调试时要先区分：

- 本来就无定义
- 还没 materialize


## 9. 最常用 API

### 9.1 构造与枚举

```julia
model = Bnc(...)
find_all_regimes!(model)
find_catalysis_regimes!(model)
match_regimes!(model)
```


### 9.2 取 regime 和条件

```julia
rgm = get_regime(model, 1)
cat = get_catalysis_regime(model, 1)
bnc = get_bnc_regime(model, bind_perm, cat_perm)

show_condition_x(rgm)
show_condition_qK(rgm)
show_condition_xk(cat)
show_condition_qKk(bnc)
show_condition_wKk(bnc)
```


### 9.3 图、路径、体积

```julia
grh = get_regimes_graph!(model; full=true)
simo = SIMOPaths(model, 1)
polys = get_polyhedra(simo)
vols = get_volumes(simo)
```


## 10. 如果我要改功能，先看哪里

- 改 binding 数学对象：`src/regimes.jl`
- 改 catalysis regime：`src/Catalysis_regime.jl`
- 改 mixed consistency：`src/Bnc_regime.jl`, `src/mixed_regime/`
- 改 `x ↔ qK` 数值求解：`src/qK_x_mapping.jl`
- 改 graph / path：`src/regime_graphs.jl`, `src/SIMO.jl`, `src/simo/`
- 改 polyhedron backend：`src/utils/poly_backend_utils.jl`, `src/utils/poly_utils.jl`, `src/simo/polyhedra.jl`
- 改 symbolic 输出：`src/symbolics.jl`, `src/output/`
- 改可视化：`src/visualize.jl`, `src/visualization/`
- 改旧 notebook 兼容：`src/old_api.jl`


## 11. 推荐阅读顺序

如果是第一次维护这个仓库，建议按这个顺序：

1. `README.md`
2. `Examples/Minimal_example.ipynb`
3. `src/BindingAndCatalysis.jl`
4. `src/regimes.jl`
5. `src/Mathcore/perm_graph_core.jl`
6. `src/SIMO.jl`
7. `src/Catalysis_regime.jl`
8. `src/Bnc_regime.jl`
9. `src/utils/poly_backend_utils.jl`
10. `test/runtests.jl`


## 12. 对维护者最有价值的测试

- `test/runtests.jl`
  主流程回归。

- `test/simo/workflows.jl`
  路径枚举、bulk path condition 和 `SIMO` 工作流回归。

- `Examples/Minimal_example.ipynb`
  最小交互式 smoke。

- `test/support/setup.jl`
  维护测试时最常用的模型工厂。

<!-- - `noback/singular_path_condition_exploration.md`
  一份近期的探索性结论，记录了 singular regime 从 path 中删除时图结构与几何条件之间的差异。 -->


## 13. 总结

这个包的核心不是“某个矩阵公式”，而是下面这条数据流：

```text
Bnc
 -> BindRegime / CatalysisRegime
 -> BncRegime
 -> RegimeGraph / SIMOPaths
 -> polyhedron / volume / symbolic / stability outputs
```

只要把下面 5 个对象和它们的边界抓住，基本不会迷路：

- `Bnc`
- `BindRegime`
- `CatalysisRegime`
- `BncRegime`
- `Polyhedra` / `CDDLib` backend glue
