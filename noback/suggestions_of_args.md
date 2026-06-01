# kwargs 传参与命名建议

日期：2026-06-01

这份文档重点审查项目中 keyword arguments 的命名是否直觉、是否符合 Julia 用户习惯、是否会让用户误解函数行为。`return_idx` 只是其中一个例子；更重要的是建立一套统一的 kwarg 命名规则，让 API 读起来稳定、可预测、容易维护。

## 总体判断

当前 API 很适合探索式使用，但 kwargs 风格还不够统一。主要问题不是某一个参数名错了，而是同一类语义在不同函数中有时用不同表达，或者一个短 kwarg 同时承担了“筛选、计算、返回形态、缓存刷新”等多层含义。

建议采用下面这条总原则：

- kwarg 名字必须说明“它控制什么维度”：输入空间、输出空间、筛选条件、计算策略、缓存策略、显示策略、solver 后端等。
- Boolean kwarg 只用于开关一个明确行为，不宜隐藏复杂模式选择。
- 只要 kwarg 会显著改变返回值类型、计算成本或副作用，就应该用更明确的名字，或者拆成独立函数。
- public API 不应大量暴露 `args...; kwargs...`；这会让用户不知道哪些参数可用，也会让 typo 很晚才报错。

## 命名优先级

我建议以后所有新增 kwargs 按下面的优先级命名：

1. 优先用完整英文词，不用缩写：`index` 优于 `idx`，`recompute` 优于 `recalc`。
2. Boolean kwarg 用动词或形容词短语：`compute_volume`, `show_volume`。坐标模式这类多选语义使用 Symbol。
3. 多选模式用 Symbol：`method=:homotopy`, `sampler=:uniform_box`, `chart=:qK`。
4. 筛选条件用 tri-state：`singular=true/false/nothing`，`asymptotic=true/false/nothing`。
5. 转发给外部库的 kwargs 必须在 docstring 里说明转发目标。

## 返回形态相关 kwargs

### `return_idx`

`return_idx::Bool` 的含义清楚，老用户也容易理解。但从 API 规范角度，它有两个弱点：

- `idx` 是缩写，新用户不如 `index` 容易理解。
- 这个 kwarg 会改变返回值类型，例如从 regime/permutation 变成 integer index。

建议：

- 兼容层保留 `return_idx=false`。
- 新 public API 尽量不要继续扩散这个模式。
- 对高频操作增加显式函数名，例如 `get_binding_indices(...)`, `get_neighbor_indices(...)`, `assign_regime_index(...)`。

不建议新增 `return_as`。虽然 `return_as=:index` 比 `return_idx=true` 更可扩展，但它仍然把返回形态藏在 kwarg 里，不如显式函数清楚。

结论：`return_idx` 只作为旧 API 兼容入口保留。新 API 用“函数名说明返回什么”。

### `return_mask`

`return_mask` 比 `return_idx` 更可接受，因为 mask 通常是附加诊断结果。但如果 `return_mask=true` 时返回 `(selected, mask)`，它仍然改变返回形态。

建议：

```julia
filter_regimes(...)              # 返回 selected
filter_regimes_mask(...)         # 返回 mask
filter_regimes_with_mask(...)    # 返回 selected, mask
```

内部和 public API 都建议逐步改成显式函数，减少 tuple 返回形态由 Bool kwarg 决定。

### `return_code`

`is_stable(rgm; return_code=true)` 不太符合 Julia 直觉。`is_*` 函数应该返回 `Bool`。如果需要 code，建议单独函数：

```julia
is_stable(rgm)        # Union{Bool,Missing}
stability_code(rgm)   # Int
```

这里允许 `missing`，因为稳定性算法可能真实地无法判断。不要把 unknown 伪装成 false。

## 输入/输出空间相关 kwargs

### `input_logspace` / `output_logspace`

这两个名字是清楚的，但它们把“坐标空间模式”表达成了两个 Bool。新 API 已改成 Symbol 模式：

```julia
qK2x(model, qK; input=:log, output=:log)
x2qK(model, x; input=:log, output=:linear)
```

旧的 `input_logspace` / `output_logspace` 仅作为兼容入口翻译，不在内部继续传播。不要再新增同义变体，例如：

- `log_input`
- `input_is_log`
- `is_log`
- `use_log`

这样比 `input_logspace=true, output_logspace=false` 更容易读，也方便未来扩展其它坐标模式。

### `log_space`

`log_space` 适合展示/符号表达函数，例如：

```julia
show_condition_qK(...; log_space=false)
show_equilibrium(...; log_space=true)
```

建议把规则固定为：

- 数值输入输出：用 `input` / `output`
- 展示表达式：用 `log_space`

这样用户能从名字判断这个 kwarg 是控制“数据坐标”还是“显示形式”。

## 筛选条件相关 kwargs

### `singular`, `asymptotic`, `feasible`

这些名字本身是好的，适合 collection/filter API：

```julia
get_regimes(model; singular=false)
get_regimes(model; asymptotic=true)
get_bnc_regimes(model; feasible=true)
```

建议明确 tri-state 语义：

- `true`：只保留满足条件的对象
- `false`：只保留不满足条件的对象
- `nothing`：不按该条件筛选

需要注意：如果 `singular` 支持 integer threshold，这必须在 docstring 明确说明，否则用户会以为只有 Bool/nothing。

### `asymptotic_only`

`asymptotic_only` 更适合 search/assignment 函数，例如：

```julia
assign_regime_qK(...; asymptotic_only=false)
assign_regime_x(...; asymptotic_only=true)
```

它表达的是“搜索范围只限 asymptotic regimes”，不是普通筛选。因此建议：

- `get_*` / `filter_*` 函数使用 `asymptotic`
- `assign_*` / search 函数使用 `asymptotic_only`
- 不要在同一个函数里同时出现 `asymptotic` 和 `asymptotic_only`

如果想更直白，可以改成 `restrict_to_asymptotic`，但太长。当前 `asymptotic_only` 可以接受。

## 计算成本与缓存相关 kwargs

### `compute_volume`

`summary_regime(...; compute_volume=false)` 是好的名字。它明确表示允许函数做额外昂贵计算。

建议保留，并把这个风格推广到其他 summary/display 函数：

```julia
summary_regime(rgm; compute_volume=false)
summary(paths; compute_volume=false)
```

### `show_volume`

`show_volume` 控制显示很自然，但不应该暗中触发重计算。如果某个函数里 `show_volume=true` 会调用 Monte Carlo volume，那名字就不够准确。

建议区分：

```julia
show_volume=true       # 是否打印/展示 volume 字段
compute_volume=false   # 是否允许昂贵计算
```

### `recompute`

`recompute` 用户能懂，但从缓存语义看，`recompute` 更短也更准确。

建议：

- 现有 API 保留 `recompute`。
- 新 API 可以考虑用 `recompute=false`。
- 不建议用 `refresh`，它更像重新读取外部状态。
- `force` 很常见，但需要说明 force 的对象，不如 `recompute` 具体。

推荐顺序：

```julia
recompute=false       # 推荐用于新 API
recompute=false     # 当前可保留
force=false           # 可用但语义较泛
refresh=false         # 不推荐用于数学计算
```

### `check`

`check=false` 很常见，但太泛。用户不知道它检查的是：

- index 是否越界
- regime 是否存在
- feasibility
- cache 是否已构建
- 是否触发 expensive computation

建议把 `check` 的语义收窄：

```julia
get_regime(model, idx; check=false)
```

只表示轻量 validation。不要让 `check=true` 触发大规模构造或昂贵计算。如果需要构造，使用显式函数：

```julia
ensure_binding_regimes!(model)
ensure_catalysis_regimes!(model)
ensure_bnc_regimes!(model)
```

如果要改名，`validate=false` 比 `check=false` 更具体。

## 算法与 solver 相关 kwargs

### `method`

`method` 适合选择项目内部算法分支：

```julia
qK2x(model, qK; method=:homotopy)
qK2x(model, qK; method=:free_energy)
qK2x(model, qK; method=:regime)
```

建议保留。

### `alg`

`alg` 适合传给 SciML/ODE solver：

```julia
x_traj_with_qK_change(...; alg=ODE.Tsit5())
```

这是 SciML 生态常见命名，可以保留。

### `solver`

如果未来引入 `solver`，建议只用来选择后端或外部 solver 类型，例如 `:ode`, `:nlsolve`, `:jump`。不要让 `method`、`alg`、`solver` 三者混用。

建议规则：

- `method`：项目内部数学算法
- `alg`：外部 solver 的算法对象
- `solver`：外部 backend 或 solver family

## 数值容差与采样相关 kwargs

当前体积和数值函数里有类似：

```julia
reltol
abstol
time_limit
batch_size
sampler
log_lower
log_upper
```

这些名字整体是清楚的。建议统一几点：

- 容差统一用 `reltol` / `abstol`，不要混用 `rtol` / `atol`，除非是直接转发给 SciML。
- 如果直接转发给 SciML，则使用 SciML 习惯的 `reltol` / `abstol`。
- 采样器用 `sampler=:gaussian` / `sampler=:uniform_box` 很直觉。
- `log_lower` / `log_upper` 清楚表达是 log-space box bounds，建议保留。

本项目应统一使用 SciML 风格的 `reltol` / `abstol`，不再保留 `rel_tol` / `abs_tol`。

## 图和可视化相关 kwargs

可视化函数保留 `kwargs...` 是合理的，因为 Makie 用户习惯传大量绘图属性：

```julia
plot_binding_regime_partition(...; colormap=:Pastel1_9, kwargs...)
draw_graph(...; node_size=..., kwargs...)
```

建议：

- 本项目自己的绘图语义用显式 kwarg，例如 `chart`, `fixed`, `ranges`, `n`, `categorical`。
- 传给 Makie 的属性继续用 `kwargs...`。
- docstring 写清楚 `kwargs...` forwarded to `heatmap!` / `meshscatter!` / `graphplot!`。

可视化 kwarg 命名一般没问题，重点是不要把数学计算 kwarg 和 Makie 样式 kwarg 混在一起。

## `args...; kwargs...` 的边界

当前很多 wrapper 都是：

```julia
foo(args...; kwargs...) = bar(args...; kwargs...)
```

这会让 API 很方便，但对用户不透明。建议分层处理：

### 可以保留 `kwargs...` 的位置

- legacy/deprecation wrapper
- Makie plotting wrapper
- ODE / NonlinearSolve 等外部 solver wrapper
- 内部 helper 函数

### 不建议暴露宽泛 `kwargs...` 的位置

- `get_regime`
- `get_regimes`
- `get_idx`
- `get_perm`
- `assign_regime`
- `summary_regime`
- `get_volume` / `get_volumes` 的 public 层

这些核心 API 最好显式列出用户能传的 kwargs。这样 docstring、IDE 补全、错误信息都会更好。

## 具体建议表

| 当前名字 | 建议 | 优先级 | 说明 |
| --- | --- | --- | --- |
| `return_idx` | 保留兼容；新增 `*_indices` 函数 | 高 | 名字可懂，但 Boolean 改返回类型不是长期最佳 |
| `return_mask` | 改为 `filter_regimes_mask` / `filter_regimes_with_mask` | 中 | 返回 tuple 时函数名更清楚 |
| `return_code` | 改为 `stability_code(...)` | 高 | `is_*` 函数应保持 Bool 返回 |
| `check` | 限定为轻量 validation；可考虑 `validate` | 高 | 当前名字过泛 |
| `recompute` | 可保留；新 API 可用 `recompute` | 中 | `recompute` 更贴近缓存计算 |
| `compute_volume` | 保留 | 高 | 清楚表达昂贵计算 |
| `show_volume` | 只控制显示，不控制计算 | 高 | show 和 compute 需要分开 |
| `input_logspace` | 仅兼容；新 API 用 `input=:log` | 中 | 坐标模式用 Symbol 更清楚 |
| `output_logspace` | 仅兼容；新 API 用 `output=:log` | 中 | 坐标模式用 Symbol 更清楚 |
| `log_space` | 限于展示/符号表达 | 低 | 和 input/output logspace 分工明确 |
| `singular` | 保留 tri-state | 中 | 需要文档说明 Bool/nothing/integer |
| `asymptotic` | 保留 tri-state | 中 | 适合 filter |
| `asymptotic_only` | assignment/search 中保留 | 中 | 适合表达搜索限制 |
| `feasible` | 保留 tri-state | 低 | 语义明确 |
| `method` | 保留 | 低 | 适合内部算法选择 |
| `alg` | 保留 | 低 | SciML 生态习惯 |
| `sampler` | 保留 | 低 | 比 `sampling_method` 更简洁 |
| `rel_tol` / `abs_tol` | 改为 `reltol` / `abstol` | 中 | 统一 SciML 风格 |

## 推荐的 API 风格示例

更推荐：

```julia
get_binding_regimes(model; singular=false)
get_binding_indices(model; singular=false)
get_binding_perms(model; singular=false)

assign_regime(model, qK; input=:log)
assign_regime_index(model, qK; input=:log)

is_stable(rgm)
stability_code(rgm)

summary_regime(rgm; compute_volume=false)
```

可以接受但不是首选：

```julia
get_regimes(model; return_idx=false)  # 旧兼容入口
assign_regime(model, qK; return_idx=false)  # 旧兼容入口
```

不建议继续扩散：

```julia
get_regimes(model; return_idx=true)
is_stable(rgm; return_code=true)
summary_regime(rgm; show_volume=true)  # 如果它会触发计算
```

## 迁移路线

1. 不立刻删除现有 kwargs，先保持兼容。
2. 为返回形态变化大的 API 增加显式函数名，例如 `*_indices`, `*_perms`, `*_with_mask`。
3. 把 `is_stable(...; return_code=true)` 拆成 `stability_code(...)`。
4. docstring 中明确 `check`, `recompute`, `show_volume`, `compute_volume` 的语义边界。
5. 新增 public API 时避免裸 `kwargs...`，优先显式列出 kwargs。
6. 在下一次 breaking release 中再考虑弱化或移除返回类型 Boolean。
