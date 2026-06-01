# 优化传参与 Regime API 维护计划

## Summary

本轮重构目标是统一 public API 与内部调用风格，减少 `return_idx`、`return_code`、`input_logspace`、`rel_tol/abs_tol` 等历史 kwarg 带来的歧义，让函数命名、返回模式和参数语义更符合 Julia 包维护习惯。

核心原则：

- 新 API 使用显式函数名表达返回类型，不再引入 `return_as`。
- 内部代码不再使用 `return_idx=false` 这类返回模式开关。
- `get_bind_*` 迁移到完整命名 `get_binding_*`。
- 旧 API 只保留在 `old_api.jl` 中做兼容转发。
- Bool、Symbol、tri-state kwargs 分工明确。

## Key Changes

### Regime 查询 API

采用显式函数族：

- `get_binding_regime`, `get_binding_regimes`
- `get_binding_indices`, `get_binding_perms`
- `get_catalysis_regime`, `get_catalysis_regimes`
- `get_catalysis_indices`, `get_catalysis_perms`
- `get_bnc_regime`, `get_bnc_regimes`
- `get_bnc_indices`, `get_bnc_perms`

`get_regime()` 仅作为 `get_binding_regime()` 的公开 alias，文档中明确说明它不是泛化的 regime 查询器。

`return_idx=false` 只保留在 `old_api.jl` 兼容层。兼容层收到旧参数后，立即转发到对应显式函数；内部实现不再传递或判断 `return_idx`。

不新增 `return_as`。原因是显式函数更易读、更易 grep、更利于维护，也避免 Symbol 返回模式造成的类型和语义不透明。

### Kwarg 命名规范

统一采用以下规则：

- Boolean kwarg 使用动词或形容词短语：`compute_volume`, `show_volume`, `input_logspace`
- 多选模式使用 Symbol：`method=:homotopy`, `sampler=:uniform_box`, `chart=:qK`
- 筛选条件使用 tri-state：`singular=true/false/nothing`, `asymptotic=true/false/nothing`
- 重新计算统一用 `recompute`，不再用 `recalc` 或 `recompute`
- 完整单词优先：`index` 优于 `idx`

转发给外部库的 `kwargs...` 必须在 docstring 中说明转发目标。

### 输入输出坐标模式

新 API 使用：

- `input=:log` 或 `input=:linear`
- `output=:log` 或 `output=:linear`

旧参数 `input_logspace`、`output_logspace` 只作为兼容入口翻译到新参数。

展示表达式相关函数继续使用 `log_space::Bool`，因为这里是单纯的显示开关，不是输入输出坐标模式。

### Filtering 与 Stability

过滤函数统一为：

- `filter_regimes(...)` 返回 selected
- `filter_regimes_mask(...)` 返回 mask
- `filter_regimes_with_mask(...)` 返回 `(selected, mask)`

稳定性函数统一为：

- `is_stable(rgm; recompute=false)` 返回 `Union{Bool,Missing}`
- `stability_code(rgm; recompute=false)` 返回底层 code

`missing` 表示算法无法判断或状态未知，避免把 unknown 错误地解释为 false。

### Asymptotic 参数

统一约定：

- `get_*` 和 `filter_*` 使用 `asymptotic`
- `assign_*` 和 search 类函数使用 `asymptotic_only`
- 同一个函数内不同时出现 `asymptotic` 和 `asymptotic_only`

### 数值容差与采样参数

数值容差统一采用 SciML 风格：

- `reltol`
- `abstol`

不保留旧的 `reltol`、`abstol` 兼容 API。

采样相关 kwargs 也按 Julia/SciML 常见习惯命名，例如：

- `rng`
- `sampler`
- `maxiters`
- `batch_size`

## Test Plan

- 更新 regime getter 测试，覆盖 `get_binding_*`, `get_catalysis_*`, `get_bnc_*` 显式函数。
- 添加旧 API 兼容测试，确认 `return_idx=false/true` 只通过兼容入口转发。
- 更新 filtering 测试，分别覆盖 selected、mask、selected with mask 三种返回函数。
- 更新 stability 测试，覆盖 stable、unstable、unknown/missing。
- 更新坐标模式测试，确认 `input=:log`、`output=:linear` 等组合行为正确。
- 更新容差参数测试，确认新 API 使用 `reltol/abstol`，旧 `reltol/abstol` 不再作为新接口支持。

## Assumptions

- 新 public API 以完整 `binding` 命名为准，`bind` 缩写只作为旧兼容入口。
- `get_regime()` 继续存在，但只代表 binding regime。
- 不引入 `return_as`。
- `is_stable` 允许返回 `missing`，因为 unknown 是真实算法状态。
- 数值容差相关旧 API 不保留兼容层。
