# Observation During Reading Code

This note records maintainability observations found while reorganizing source layout. It is intentionally a working note, not an API commitment.

## 已在本轮顺手处理

- `src/BindingAndCatalysis.jl`
  - 顶层文件原来把 imports、exports、type definitions、include order 混在一起；已分成 dependencies、internal paths、public exports、shared types、source loading。
  - `include(joinpath(...))` 的重复写法较多；已加入 `_include_src` / `_include_mathcore`，让 include 顺序更可读。
  - `Makie` / `GraphMakie` 仍然在主模块 eager-load。这样对用户安装和 `using BindingAndCatalysis` 的成本较高，建议后续考虑把 visualization 做成 extension 或 lazy include。

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

- 术语迁移还没有完成。
  - 代码内部仍大量使用 `vertex/vertices` 命名表示 regime，例如 `vertices_graph`, `vertices_data`, `vertices_perm_dict`。
  - 这些字段属于内部结构，可逐步迁移到 `regimes_graph`, `regimes_data`, `regimes_perm_dict`，并在 legacy 层保持兼容。

- `RegimeCore.jl` 仍然过大。
  - 当前约 1000 行，混合了 network access、cache ensure、identity getter、filter、affine maps、condition matrices。
  - 建议拆成 `regime_accessors.jl`, `regime_filters.jl`, `regime_affine_maps.jl`, `regime_conditions.jl`。

- `get_C_C0_*` / `get_affine_*` wrapper 模式重复明显。
  - 很多函数只是取 tuple 的第 1/2 项，或先 `get_*_regime` 再调用具体方法。
  - 可以用少量内部 helper 统一 tuple projection 和 regime materialization，减少 API 扩展时的漏改风险。

- `summary_regime` 会做实质计算。
  - `summary_regime` 内部调用 `get_volume`，regular regime 会触发 Monte Carlo 采样。
  - 建议把展示和计算拆开，例如 `summary_regime(...; compute_volume=false)`，默认不做重计算。

- `old_api.jl` 的 deprecation wrapper 很长。
  - 现在每个 alias 都手写一行，容易漏掉 export 或目标函数。
  - 可考虑用一个小表生成 alias，或至少按 `vertex`, `mixed`, `qss/w`, `SISO/SIMO` 四类分组。

- `qK2x` solver dispatch 仍可继续拆分。
  - 现在所有 solver branch 都在一个函数内。后续可拆成 `_solve_qK2x_homotopy`, `_solve_qK2x_free_energy`, `_solve_qK2x_nullspace`, `_solve_qK2x_nlsolve`。
  - 这样会更容易测试每个 solver 的 start point 和 tolerance 行为。

- `CatalysisData` constructor 负担较重。
  - 它同时做 sparse conversion、affine k validation、left-nullspace、basis repair、binding model mutation、helper construction。
  - 后续可以分为 validation、basis derivation、binding mutation、data construction 四步，便于测试 `update_catalysis!` “初始化只跑一次”的约束。

- 可视化依赖建议延迟加载。
  - `src/BindingAndCatalysis.jl` 顶层加载 `Makie` 和 `GraphMakie`，使非绘图用户也承担编译成本。
  - 如果 Julia 版本允许，建议改为 package extension；否则可以把 visualization exports 和 includes 延迟到用户显式调用路径。

- 建议引入 formatter。
  - 当前项目环境没有 `JuliaFormatter`。如果后续加入，建议固定 `.JuliaFormatter.toml`，避免每次维护产生大面积无意义 diff。
