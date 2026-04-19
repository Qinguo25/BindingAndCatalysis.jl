# BindingAndCatalysis.jl Source Helpers And Data Structures

这份文档是对 `src/` 的维护者视角盘点，目标有 3 个：

1. 说明各模块的主要数据结构放在哪里。
2. 说明各文件里的辅助函数大致在做什么。
3. 标出目前已经出现、且值得继续复用或抽象的重复逻辑。


## 1. 总体结论

当前源码已经形成了比较明确的分层：

- 核心领域对象：`src/BindingAndCatalysis.jl`
- binding regime / graph / affine propagation：`src/regimes.jl` 与 `src/Mathcore/`
- mixed regime：`src/mixed_regime/`
- polyhedral facade 与后端：`src/PolyBackend.jl`、`src/CddBridge.jl`、`src/NativePolyhedra/`
- SIMO path workflow：`src/simo/`
- symbolic / visualization：`src/output/`、`src/visualization/`
- 通用工具：`src/utils/`

目前最值得继续抽象的重复点，不在底层多面体算法，而在业务层：

1. `mixed_regime` 中 regular / singular 条件装配
2. regime / polyhedron 的筛选与 mask 逻辑
3. regime 初始化 / materialization 生命周期
4. graph 可视化中的节点筛选与 edge label 生成
5. constraint signature / canonicalization 的统一接口

相反，下列部分当前已经比较合理，不建议再强行合并：

- `NativePolyhedra` 内部的大量低层 helper
- `CddBridge` 的 token parse / emit helper
- `old_api.jl` 的兼容别名层


## 2. 文件级盘点

### 2.1 顶层入口与 shim 文件

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/BindingAndCatalysis.jl` | `Volume`, `IntegrationHelper`, `NρCacheEntry`, `Hyperplane_perm`, `ChoiceIneq`, `MatrixHelper`, `Regimes`, `BindRegime`, `CatalysisRegime`, `BncRegime`, `CatalysisData`, `Bnc` | 模块入口；定义核心对象；少量核心 accessor 和 affine mode 规范化 | 顶层核心类型仍然集中在这里 |
| `src/helperfunctions.jl` | 无 | 只是 `src/utils/*.jl` 的聚合 shim | 不应再往这里加新逻辑 |
| `src/SIMO.jl` | 无 | 只是 `src/simo/*.jl` 的聚合 shim | 当前角色清晰 |
| `src/symbolics.jl` | 无 | 只是 `src/output/*.jl` 的聚合 shim | 当前角色清晰 |
| `src/visualize.jl` | 无 | 只是 `src/visualization/*.jl` 的聚合 shim | 当前角色清晰 |
| `src/Bnc_regime.jl` | 无 | 只是 `src/mixed_regime/*.jl` 的聚合 shim | 当前角色清晰 |


### 2.2 binding / regime 主流程

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/regimes.jl` | 无新增 struct | `_initialize_regime!`, `_materialize_qK_conditions!`, `_fill_all_info!`, `_affine_mapping_polyhedra`, `_get_mask` | binding regime 的 materialization 和筛选入口 |
| `src/regime_graphs.jl` | 无新增 struct | `_reachable_from_sources`, `_can_reach_sinks`, `_enumerate_paths`, `_ensure_full_regimes_graph!` | 图遍历与 path 枚举 |
| `src/regime_assign.jl` | `QKHyperplaneClassifier` | `_qK_signature`, `_candidate_regime_positions`, `_candidate_regimes`, `_build_qK_hyperplane_classifier`, `_assign_regime_qK_fallback` | qK 点到 regime 的分类器 |
| `src/initialize.jl` | 无 | `_change_q_L_order!`, `_remove_regime_data!`, `_rebuild_helper!` 相关流程 | 模型变换后的缓存失效与重建 |
| `src/volume_calc.jl` | 无 | `_calc_volume_via_classifier`, `_remove_poly_intersect`, `_get_mask` | regime / polyhedron volume 估计 |
| `src/qK_x_mapping.jl` | `HomotopyParams`, `TimecurveParam` | `_logqK2logx_nlsolve`, `_logx_traj_with_logqK_change` | 数值反解与轨线 |
| `src/numeric.jl` | 无 | 数值 ODE / Jacobian 更新相关 helper | 目前 helper 较少，仍偏过程式 |
| `src/old_api.jl` | 无 | 旧命名到新命名的薄别名层 | 兼容层，重复是故意的 |


### 2.3 mixed regime

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/mixed_regime/bnc_core.jl` | 无新增 struct | `_build_BncRegime`, `_binding_C_qKk` | mixed regime 容器的搭建 |
| `src/mixed_regime/bnc_initialization.jl` | 无新增 struct | `_build_row_affine_cache`, `_build_row_context`, `_init_regular_bnc_regime!`, `_calc_singular_H_ss`, `_init_singular_bnc_regime!`, `_initialize_regime!` | mixed regime 的初始化主线 |
| `src/mixed_regime/bnc_conditions.jl` | 无新增 struct | `_project_bnc_singular_condition`, `_calc_C_qKk_*` / `_calc_C_wKk_*`, `_steady_state_offsets`, `_expand_Hw_to_wKk` | mixed 条件搬运与 polyhedral projection |
| `src/mixed_regime/bnc_display.jl` | 无新增 struct | 以 display 为主，少量初始化触发 | 展示层不应承载复杂计算 |
| `src/Catalysis_regime.jl` | 无新增 struct | `_initialize_regime!(::CatalysisRegime)` | catalysis regime 的惰性 materialization |


### 2.4 Mathcore

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/Mathcore/find_matrix_vertex.jl` | 无新增 struct | `_build_matrix_helper`, `_enumerate_asymptotic_regimes`, `_enumerate_all_regimes`, `_perm_process`, `_calc_P_P0`, `_calc_C_C0` | regime 枚举的入口 |
| `src/Mathcore/matrix_inverse.jl` | 无新增 struct | `_build_regular_H_from_key_entry`, `_build_singular_H_from_perm`, `_calc_nullity`, `_calc_H`, `_adj_singular_matrix`, `_exact_direct_inverse_or_adjugate` | `H/H0` 与 nullity 的核心线代逻辑 |
| `src/Mathcore/perm_graph_core.jl` | `RegimeEdge`, `RegimeHyperplane`, `RegimeGraph` | `_calc_regimes_graph`, `_edge_qK_interface`, `_canonicalize_qK_interface`, `_intern_qK_interface!`, `_fulfill_regimes_graph!` | regime graph 的核心数据结构与构建逻辑 |
| `src/Mathcore/graph_propagate.jl` | `AffinePropagateWorkspace`, `SeedAnalysisState` | `_initialize_regular_seed_affine!`, `_prefill_affine_cache!`, `_process_component_from_seed_scan!`, `_propagate_from_regular_seed!` | affine cache propagation |
| `src/Mathcore/d_stable.jl` | 无新增 struct | `_obvious_not_hurwitz`, `_neg_lyap_triangle`, `_signed_diag_lyap_margin` | 稳定性检查 |
| `src/Mathcore/SparseSparse_modified.jl` | `luFac` | 稀疏线代辅助实现 | 独立性较强 |


### 2.5 Polyhedral facade 与后端

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/PolyBackend.jl` | 无新增 struct | `backend_eliminate`, `backend_intersect_many`, `backend_project_hrep`, `backend_from_fastpath` | `cdd/cddlog` facade，业务代码应优先经由这里 |
| `src/CddBridge.jl` | 无新增 struct | `_local_cdd_bindir`, `_cdd_numbertype`, `_polyhedron_to_C_C0_nullity`, `_write_cdd_hrep`, `_parse_cdd_hrep`, `_canonicalize_hrep` | 本地 `cdd/cddlog` bridge |
| `src/ExactTypes.jl` | `ExactLogExpr` | `_factor_positive_integer` | exact 常数类型 |
| `src/NativePolyhedra/NativePolyhedra.jl` | 无新增 struct | 模块入口与 cache invalidation stub | 薄入口 |
| `src/NativePolyhedra/polyhedra_core.jl` | `HyperPlane`, `HalfSpace`, `HRep`, `Polyhedron` | 约束转换、canonicalization、冗余删除、消元、LP 优化、集合包含 | H-rep 主实现 |
| `src/NativePolyhedra/vrep_core.jl` | `VRep` | rank/nullspace、vertex/ray enumeration、dual/project/block elimination | V-rep 与投影实现 |


### 2.6 SIMO

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/simo/core.jl` | `SIMOPaths` | `_build_paths_dict`, `_ensure_paths_dict!`, `_build_path_edge_index`, `_normalize_simo_path_selection`, `_path_indices_to_calculate` | path 容器与索引层 |
| `src/simo/polyhedra.jl` | 无新增 struct | `_ensure_node_polyhedra!`, `_ensure_edge_polyhedra!`, `_build_path_polyhedron`, `_calc_polyhedra_for_paths_bulk_suffix_dag!`, `_resolve_simo_rebase_mat` | path condition 与 path volume 的 polyhedral 计算 |
| `src/simo/reaction_order.jl` | 无新增 struct | `_calc_RO_for_single_path`, `_dedup`, `_ensure_ro_regimes_materialized!` | reaction-order 路径分析 |
| `src/simo/display.jl` | 无新增 struct | 以展示 API 为主 | 当前拆分合理 |


### 2.7 Symbolic output

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/output/symbolic_symbols.jl` | 无新增 struct | `_flux_sym` | 符号命名辅助 |
| `src/output/symbolic_renderers.jl` | 无新增 struct | `show_condition_poly`, `_exp10_factor`, `_render_condition_from` | 条件 / 表达式渲染核心 |
| `src/output/symbolic_api.jl` | 无新增 struct | 各 `show_condition_*` / `show_expression_*` 的公共入口 | 公共 API 面 |
| `src/output/symbolic_paths.jl` | `PathRow` | `_normalize_rows` | path 输出格式化 |


### 2.8 Visualization

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/visualization/graphs.jl` | `RegimeColorMap` | `_edge_interface_label`, `_default_graph_layout`, `_resolve_graph_layout`, `_filter_edge_labels_for_nodes`, `_materialize_node_sizes` | regime / path graph 绘图主线 |
| `src/visualization/simo_plot.jl` | 无新增 struct | SIMO path plots | 已与 graph plot 分离 |
| `src/visualization/rop.jl` | 无新增 struct | `_lock_current_limits!`, `_rop_axis_label` | reaction-order plots |
| `src/visualization/poly_slices.jl` | 无新增 struct | `_grid_sample_polyhedron` | polyhedron slice sampling |


### 2.9 utils

| 文件 | 主要数据结构 | 辅助函数 / 角色 | 备注 |
| --- | --- | --- | --- |
| `src/utils/matrix_utils.jl` | 无新增 struct | `L_from_N`, `N_from_L`, `rowmask_indices`, `diag_indices`, `rebase_mat_lgK`, `_Mtx2idx_val`, `_idx_val2Mtx` | 线代/矩阵小工具 |
| `src/utils/model_utils.jl` | 无新增 struct | `randomize`, `N_generator`, `L_generator`, `locate_sym*` | 模型构造与 symbol lookup |
| `src/utils/graph_utils.jl` | 无新增 struct | `graph_from_paths`, `sources_sinks_from_paths`, `vector_difference`, `compress_adjacency` | 图辅助 |
| `src/utils/symbolic_utils.jl` | 无新增 struct | `name_converter`, `log10_sym`, `exp10_sym`, `render_array`, `strip_before_bracket` | 符号渲染小工具 |
| `src/utils/poly_utils.jl` | 无新增 struct | `_normalized_constraint_signature`, `_hrep_signature`, `same_polyhedron` | polyhedron 比较辅助 |
| `src/utils/misc_utils.jl` | 无新增 struct | `arr_to_vector`, `pythonprint`, `_ode_solution_wrapper` | 低耦合杂项工具 |


## 3. 已经做得比较好的复用点

这些点已经值得保留，不建议再打散：

### 3.1 qK hyperplane canonicalization 已统一

`src/Mathcore/perm_graph_core.jl` 里的 `_canonicalize_qK_interface` 现在同时被：

- regime graph 构建
- `QKHyperplaneClassifier`

复用。这个方向是对的。说明“超平面去重 / 定向 / exact-vs-float key”这层逻辑已经开始集中，不要再复制第三套。


### 3.2 SIMO 路径选择归一化已统一

`src/simo/core.jl` 里的：

- `_normalize_simo_path_selection`
- `_path_indices_to_calculate`

已经把 `get_polyhedra`、`get_volumes`、`get_RO_paths` 里的路径索引归一化收口了。这是很好的模式，后续别回到每个 API 自己处理 `pth_idx`。


### 3.3 顶层 shim 已经把聚合层和实现层拆开

`helperfunctions.jl`、`SIMO.jl`、`symbolics.jl`、`visualize.jl`、`Bnc_regime.jl` 当前都是“纯聚合入口”，这比早期把所有逻辑继续堆在大文件里要健康得多。


## 4. 目前最值得继续抽象的重复逻辑

### 4.1 mixed_regime 条件装配存在明显重复

位置：

- `src/mixed_regime/bnc_conditions.jl`

目前至少存在 3 组 regular / singular 双分支：

- `_calc_C_qKk_catalysis_only_regular/singular`
- `_calc_C_qKk_cat_regular/singular`
- `_calc_C_qKk_ss_regular/singular`

重复模式是：

1. 组装 block matrix
2. 拼 `C0`
3. 指定 equality 行数
4. 指定需要消去的变量轴
5. 调 `_project_bnc_singular_condition`

建议抽象：

- 一个内部 builder，例如 `assemble_projection_problem(...)`
- 或一个更显式的 block DSL，例如
  `hcat_blocks`, `vcat_blocks`, `projection_problem(C, C0; neq, delset)`

收益：

- mixed regular/singular 逻辑会更容易对照
- exact / float 路径更容易共同维护
- 后续如果 projection 后端再改，不需要改 3 组拼装代码


### 4.2 regime / polyhedron 过滤逻辑有重复语义

位置：

- `src/regimes.jl` 的 `_get_mask(::AbstractVector{<:BindRegime})`
- `src/volume_calc.jl` 的 `_get_mask(::AbstractVector{<:Polyhedron})`
- `src/simo/reaction_order.jl` 通过 `_get_mask(model, path; ...)` 间接复用 regime mask

这些函数都在表达“按 singular / asymptotic 条件做筛选”，但：

- 绑定到不同对象类型
- 谓词构造重复
- 语义细节略有漂移

建议抽象：

- 抽一个统一谓词构造器，例如 `make_regime_filter(; singular, asymptotic)`
- `BindRegime`、`Polyhedron` 仅各自提供：
  - `nullity_like(x)`
  - `asymptotic_like(x)`

收益：

- 筛选语义只维护一份
- volume / regime / RO path 的筛选结果更容易保持一致


### 4.3 regime 初始化生命周期仍有多处手工触发

位置：

- `src/regimes.jl`
- `src/Catalysis_regime.jl`
- `src/mixed_regime/bnc_initialization.jl`
- `src/simo/reaction_order.jl`

目前存在多种不同层级的“确保数据已经 materialize”的调用：

- `_initialize_regime!`
- `_materialize_qK_conditions!`
- `_fill_all_info!`
- `_ensure_ro_regimes_materialized!`

这不是错误，但比较容易出现：

- 某个路径少调用一步
- display 层顺便触发重计算
- mixed / SIMO 层各自维护自己的预热流程

建议抽象：

- 一个统一内部协议，例如 `ensure_regime_ready!(rgm; affine=false, qK=false, full=false)`
- 再通过多重派发区分 `BindRegime` / `CatalysisRegime` / `BncRegime`

收益：

- 生命周期更显式
- 更容易做 lazy / eager 切换
- 更容易在测试中覆盖“状态未 materialize”边界


### 4.4 graph 可视化里的节点筛选和 edge label 逻辑还可继续复用

位置：

- `src/visualization/graphs.jl`

目前已经有：

- `_node_subset_by_nullity`
- `_hide_isolated_nodes`
- `_filter_edge_labels_for_nodes`
- `_edge_interface_label`

这些逻辑本身没错，但都围绕同一个问题：

- “当前视图中哪些节点保留”
- “保留后如何同步 node label / edge label / edge style”

建议抽象：

- 一个 `graph_view(grh; keep_nodes=...)` 风格 helper
- 统一返回：
  - 压缩后的 graph
  - old→new index map
  - node subset
  - 已过滤的 edge labels

收益：

- `draw_graph` 会更短
- 2D/3D、Bnc/SIMO 共用更多逻辑
- 后续如果再加“隐藏某类 regime”选项，改动面更小


### 4.5 constraint signature 逻辑还可以继续集中

位置：

- `src/utils/poly_utils.jl`
- `src/NativePolyhedra/polyhedra_core.jl`

当前：

- `NativePolyhedra` 内部有 `_constraint_signature`
- `utils/poly_utils.jl` 又有 `_normalized_constraint_signature` / `_hrep_signature`

两边不是完全重复，但都在做“把 H-rep 约束变成可比较的 canonical signature”。

建议抽象：

- 让 `NativePolyhedra` 暴露一个更稳定的内部 canonical-signature helper
- `same_polyhedron` 直接复用那一套，而不是在 `utils/poly_utils.jl` 再拼第二套签名规则

收益：

- polyhedral canonicalization 规则更统一
- exact / float / signed-vs-unsigned 等细节不容易分叉


## 5. 不建议现在继续抽象的部分

### 5.1 CddBridge

`src/CddBridge.jl` 里 helper 很多，但大多是同一条 I/O pipeline 上的细小步骤：

- numbertype 判断
- token 编码
- H-rep 文件写出
- tool 调用
- H-rep 解析

它们数量多，但凝聚度高。当前更像“桥接协议实现”，不是重复业务逻辑。这里不建议为“减少函数数目”再进一步折叠。


### 5.2 NativePolyhedra

`src/NativePolyhedra/polyhedra_core.jl` 和 `src/NativePolyhedra/vrep_core.jl` helper 极多，但这是低层算法模块的正常现象。  
这里最重要的是：

- 约束表示一致
- cache invalidation 正确
- eliminate / canonicalize / vrep 转换正确

不建议为了“看起来少几个 helper”而硬合并函数。


### 5.3 old_api

`src/old_api.jl` 的重复命名是兼容层，本身就是故意的。除非决定整体删旧 API，否则不需要重构。


## 6. 建议的后续重构顺序

如果按收益 / 风险比排序，我建议这样做：

1. 先抽 `mixed_regime` 的 projection problem builder
2. 再统一 regime / polyhedron filter predicate
3. 然后把 regime materialization 生命周期抽成统一内部协议
4. 最后整理 graph view / edge label 的可视化辅助层

原因：

- 前两项会直接减少重复代码和分支复杂度
- 第三项能降低“某条路径忘了初始化”的维护风险
- 第四项主要提升可读性和可扩展性，不像前两项那样直接影响 correctness


## 7. 一个简短判断

当前代码结构的主要问题，已经不再是“文件太乱”，而是：

- 领域逻辑已经拆开，但业务层仍存在若干“同义 block assembly / 同义 filter / 同义 lifecycle”分散实现

也就是说，下一阶段最值的工作不是继续拆文件，而是：

- 在已经分层的基础上，继续抽掉重复的业务模式。
