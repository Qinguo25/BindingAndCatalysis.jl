# Observation During Reading Code

This note records maintainability observations found while reorganizing source layout. It is intentionally a working note, not an API commitment.

## 已在本轮顺手处理

- `src/BindingAndCatalysis.jl`
  - 顶层文件原来把 imports、exports、type definitions、include order 混在一起；已分成 dependencies、internal paths、public exports、shared types、source loading。
  - `include(joinpath(...))` 的重复写法较多；已加入 `_include_src` / `_include_mathcore`，让 include 顺序更可读。
  - `Makie` / `GraphMakie` 已从主模块 eager-load 移到 package extension。当前实现为了复用现有 visualization 文件，会在 extension 加载时把这些文件 include 到父模块；后续如果继续拆分，可把 color map/helper 与 GraphMakie graph 绘制分开，让纯 Makie plot 不必依赖 GraphMakie。

- `src/initialize.jl`
  - 文件现在分成 construction、qK2x method selection、catalysis initialization、conservation basis repair、cache helpers、display helpers。
  - `summary(model::Bnc)` 和 `show(io, MIME"text/plain", model)` 原来有大段重复输出逻辑；已合并到 `_print_bnc_summary`。
  - `new_ord !== collect(1:length(new_ord))` 是对象 identity 比较，语义上应该是值比较；已改成 `new_ord != collect(...)`，避免相同顺序也被当成重排。

- `src/RegimeCore.jl`
  - 顶部 getter/ensure/predicate/filter/affine sections 已重新分块，部分 spacing 已整理。
  - `get_nullity(rgm::CatalysisRegime)` 目前返回 `r_v`，这更像 catalysis flux-balance nullity/constraint rank，不一定是用户理解的 regime nullity。建议后续明确命名或补充文档。

- `src/qK_x_mapping.jl`
  - `x2qK` 原来有四层 `input_logspace`/`output_logspace` 重复分支；已改为统一 `logx` 和 `x_linear` 流程。
  - `qK2x` 的 solver dispatch 分块和缩进已整理。

- tests
  - SIMO workflow 默认测试已经换成较小模型，重的 polyhedra equality 和 plotting 检查用环境变量开启。

## 建议后续单独处理

- 术语迁移仍有遗留字段。
  - `Regimes` 已迁移到 `regimes_data` / `regimes_perm_dict`，并保留旧 `vertices_*` 读取兼容。
  - `vertices_graph` 和 `_vertices_Nρ_inv_dict` 仍是结构体字段名。直接改字段会破坏序列化/外部访问，建议在下一个 breaking release 迁移到 `regimes_graph` 和 `_regimes_Nρ_inv_dict`。

- `RegimeCore.jl` 仍然过大。
  - 当前约 1000 行，混合了 network access、cache ensure、identity getter、filter、affine maps、condition matrices。
  - 建议拆成 `regime_accessors.jl`, `regime_filters.jl`, `regime_affine_maps.jl`, `regime_conditions.jl`。

- `get_C_C0_*` / `get_affine_*` wrapper 模式重复明显。
  - 很多函数只是取 tuple 的第 1/2 项，或先 `get_*_regime` 再调用具体方法。
  - 可以用少量内部 helper 统一 tuple projection 和 regime materialization，减少 API 扩展时的漏改风险。

- `summary_regime` 已改为默认不计算 volume。
  - 当前 `summary_regime(...; compute_volume=false)` 只打印提示；显式 `compute_volume=true` 才会调用 `get_volume`。
  - 后续可考虑增加 `regime_summary(...) -> NamedTuple`，让打印和数据收集进一步分离。

- `old_api.jl` 的 deprecation wrapper 很长。
  - 现在每个 alias 都手写一行，容易漏掉 export 或目标函数。
  - 可考虑用一个小表生成 alias，或至少按 `vertex`, `mixed`, `qss/w`, `SISO/SIMO` 四类分组。

- `qK2x` solver dispatch 仍可继续拆分。
  - 现在所有 solver branch 都在一个函数内。后续可拆成 `_solve_qK2x_homotopy`, `_solve_qK2x_free_energy`, `_solve_qK2x_nullspace`, `_solve_qK2x_nlsolve`。
  - 这样会更容易测试每个 solver 的 start point 和 tolerance 行为。

- `CatalysisData` constructor 负担较重。
  - 它同时做 sparse conversion、affine k validation、left-nullspace、basis repair、binding model mutation、helper construction。
  - 后续可以分为 validation、basis derivation、binding mutation、data construction 四步，便于测试 `update_catalysis!` “初始化只跑一次”的约束。

- 可视化 extension 后续还可继续细分。
  - 当前 extension 以 `Makie + GraphMakie` 为触发条件，覆盖 SIMO plot、graph plot、ROP、partition plot 和 polyhedron slices。
  - `plot_binding_regime_partition` 等纯 Makie 图理论上不需要 GraphMakie；可以后续拆成 Makie-only extension 和 GraphMakie extension。

- Formatter 已引入但未做全仓库重排。
  - 已添加 `.JuliaFormatter.toml` 和 `scripts/format` 独立 formatter 环境。
  - 由于当前任务包含功能性改动，暂未运行全仓库格式化，避免把语义 diff 和纯格式 diff 混在一起。
