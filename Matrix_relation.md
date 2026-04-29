# Matrix Relation

这份文档把项目里最常见、也最容易混淆的矩阵关系集中放在一处。建议和 [Archetecture.md](/home/joker/Realizibility_index/BindingAndCatalysis.jl/Archetecture.md) 一起看。

## 1. 变量与基底

- `x ∈ R^n_{>0}`: free species
- `q = (q_cat, q_dep, q_para) ∈ R^d_{>0}`: total concentrations
- `K ∈ R^r_{>0}`: binding equilibrium constants
- `k ∈ R^{n_v}_{>0}`: catalytic rate constants
- `q_c = (q_cat, w, q_para)`: catalysis-constrained total coordinates
- `q_ss = (w, q_para)`: steady-state reduced coordinates

最常用的坐标基底有 3 个：

- `(x, k)`
- `(q_c, K, k)`，代码里通常直接写成 `(q, K, k)`，其中 `q` 已按 `(q_cat, w, q_para)` 排好
- `(q_ss, K, k)`


## 2. 网络层矩阵

### 2.1 Binding

```math
q = Lx,
\qquad
N \log x = \log K.
```

- `L ∈ R^{d×n}`: total map
- `N ∈ R^{r×n}`: equilibrium-constraint matrix
- `d + r = n`

### 2.2 Catalysis

```math
\dot q_{cat,dep} = \Gamma v,
\qquad
\log v = \Pi \log x + \log k.
```

- `Γ ∈ R^{(d_cat+d_dep)×n_v}`: catalytic change matrix
- `Π ∈ R^{n_v×n}`: flux exponent matrix

取 `Γ` 的左零空间基：

```math
L_\Gamma \Gamma = 0.
```

定义：

```math
w := L_\Gamma q_{cat,dep},
\qquad
L_w := L_\Gamma L_{cat,dep}.
```

再取 `Γ` 的 full-row-rank 活跃部分：

```math
T \Gamma =
\begin{bmatrix}
S\\
0
\end{bmatrix},
\qquad
T =
\begin{bmatrix}
I_{d_cat} & 0\\
L_\Gamma
\end{bmatrix}.
```

于是 reduced dynamics 为

```math
\dot q_{cat} = Sv,
\qquad
\dot w = 0,
\qquad
\dot q_{para} = 0.
```


## 3. Binding regime 矩阵

对一个固定 binding dominance perm，

```math
f(L) = (P_0, P, C_{0,x}, C_x).
```

含义是：

```math
\log q = P \log x + P_0,
\qquad
C_x \log x + C_{0,x} \ge 0.
```

再与 binding equilibrium 合并：

```math
M :=
\begin{bmatrix}
P\\
N
\end{bmatrix},
\qquad
M_0 :=
\begin{bmatrix}
P_0\\
0
\end{bmatrix}.
```

所以：

```math
\log(q,K) = M \log x + M_0.
```

若 `M` regular，则

```math
H := M^{-1},
\qquad
H_0 := -HM_0,
\qquad
\log x = H \log(q,K) + H_0.
```

把 `x`-space dominance 搬到 `(q,K)`：

```math
C_{qK} = C_x H,
\qquad
C_{0,qK} = C_x H_0 + C_{0,x}.
```


## 4. Catalysis regime 矩阵

### 4.1 从 `S` 到 `[S^+; S^-]`

代码里先把 `S` 拆成正负两部分并按行堆叠：

```math
S_{pos\_neg} :=
\begin{bmatrix}
S^+\\
S^-
\end{bmatrix}.
```

注意这里 `S^+ v`、`S^- v` 一般是正线性映射，不只是 selector。

### 4.2 Regime representation

对固定 catalysis perm，

```math
f(S_{pos\_neg}) =
\left(
\begin{bmatrix}P_0^+\\P_0^-\end{bmatrix},
\begin{bmatrix}P^+\\P^-\end{bmatrix},
\begin{bmatrix}C_0^+\\C_0^-\end{bmatrix},
\begin{bmatrix}C^+\\C^-\end{bmatrix}
\right).
```

代码字段对应：

- `P_pos_neg`, `P0_pos_neg`
- `C`, `C0`

再定义差分 steady-state 量：

```math
P^\theta := P^+ - P^-,
\qquad
P_0^\theta := P_0^+ - P_0^-.
```

代码字段：

- `P`
- `P0`

于是在 `v` 基底：

```math
P^\theta \log v + P_0^\theta = 0,
\qquad
C^\theta \log v + C_0^\theta \gg 0.
```

搬到 `(x,k)` 基底后：

```math
P_\Pi := P^\theta \Pi,
\qquad
C_\Pi := C^\theta \Pi,
```

```math
P^\theta \Pi \log x + P^\theta \log k + P_0^\theta = 0,
```

```math
C^\theta \Pi \log x + C^\theta \log k + C_0^\theta \gg 0.
```

代码字段：

- `PΠ`
- `CΠ`


## 5. Mixed regime 矩阵

### 5.1 Catalysis-constrained binding map

定义

```math
q_c := (q_cat, w, q_para),
\qquad
L_c :=
\begin{bmatrix}
L_{cat}\\
L_w\\
L_{para}
\end{bmatrix}.
```

对固定 binding regime，

```math
\log q_c = P \log x + P_0,
\qquad
\log(q_c, K) = M \log x + M_0.
```

### 5.2 Mixed consistency in `(q,K,k)`

regular binding regime 下：

```math
\log x = H \log(q,K) + H_0.
```

binding condition:

```math
C_{qK}\log(q,K) + C_{0,qK} \ge 0.
```

catalysis dominance:

```math
C^\theta \Pi \log x + C^\theta \log k + C_0^\theta \gg 0.
```

代入后：

```math
C_{qKk}^{cat} =
\begin{bmatrix}
C_{qK} & 0\\
C_\Pi H & C^\theta
\end{bmatrix},
\qquad
C_{0,qKk}^{cat} =
\begin{bmatrix}
C_{0,qK}\\
C_\Pi H_0 + C_0^\theta
\end{bmatrix}.
```

singular binding regime 下，代码改为在扩展变量

```text
(\log(q,K), \log k, \log x)
```

上做 elimination。


## 6. Steady-state reduction

定义：

```math
q_{ss} := (w, q_{para}),
\qquad
L_{ss} :=
\begin{bmatrix}
L_w\\
L_{para}
\end{bmatrix},
```

```math
N_{ss} :=
\begin{bmatrix}
N\\
P^\theta \Pi
\end{bmatrix},
\qquad
\log K_{ss} :=
\begin{bmatrix}
\log K\\
-(P^\theta \log k + P_0^\theta)
\end{bmatrix}.
```

对固定 reduced binding regime，

```math
\log q_{ss} = P_{ss} \log x + P_{0,ss}.
```

因此：

```math
M_{ss} :=
\begin{bmatrix}
P_{ss}\\
N_{ss}
\end{bmatrix},
\qquad
M_{0,ss} :=
\begin{bmatrix}
P_{0,ss}\\
0
\end{bmatrix},
```

```math
\log(q_{ss}, K_{ss}) = M_{ss} \log x + M_{0,ss}.
```

若 `M_ss` regular，则

```math
H_{ss} := M_{ss}^{-1},
\qquad
H_{0,ss} := -H_{ss} M_{0,ss}.
```

把 `K_ss` 展开回 `(K,k)`：

```math
H_{ss} = [H_L \;\; H_R],
```

其中 `H_R` 对应最后 `d_cat` 行的 `K_ss` 分量，则

```math
H_{ssk} = [H_L \;\; -H_R P^\theta],
\qquad
H_{0,ssk} = H_{0,ss} - H_R P_0^\theta.
```

所以最终：

```math
\log x = H_{ssk} \log(q_{ss}, K, k) + H_{0,ssk}.
```

这正是代码里 `BncRegime.H` / `BncRegime.H0` 的 `(q_ss, K, k) -> x` 版本。


## 7. `q_cat` 的显式表达

一旦 `BncRegime` regular，就可以把 binding regime 前 `r_v` 行取出来：

```math
\log q_{cat} = P_{cat} \log x + P_{0,cat}.
```

代入 `x = H_{ssk} (q_{ss}, K, k) + H_{0,ssk}` 后：

```math
F := P_{cat} H_{ssk},
\qquad
F_0 := P_{0,cat} + P_{cat} H_{0,ssk},
```

```math
\log q_{cat} = F \log(q_{ss}, K, k) + F_0.
```


## 8. Stability screening matrix

对 mixed regime，

```math
H^{bd} := P^\theta \Pi H \pi,
```

其中 `H` 是原始 binding regime 的 `(q,K) -> x` 映射，`π` 选出 `q_cat` 对应列。

它用于 regime-level diagonal stability screening：

- `H_bd` 只依赖 binding affine sensitivity 和 catalysis steady-state exponents
- `P_0^\theta`、`C_0^\theta` 不进入 `H^{bd}`，因为对 `\log q_cat` 求导时常数项会消失


## 9. 代码里最常见的字段对照

- `BindRegime.P`, `BindRegime.P0`: `q` 对 `x` 的 dominant monomial map
- `BindRegime.M`, `BindRegime.M0`: `(q,K)` 对 `x` 的 affine map
- `BindRegime.H`, `BindRegime.H0`: `(q,K) -> x`
- `BindRegime.C_x`, `BindRegime.C0_x`: binding dominance in `x`
- `BindRegime.C_qK`, `BindRegime.C0_qK`: binding dominance in `(q,K)`
- `CatalysisRegime.P_pos_neg`, `CatalysisRegime.P0_pos_neg`: raw `f([S^+;S^-])`
- `CatalysisRegime.P`, `CatalysisRegime.P0`: `P^\theta`, `P_0^\theta`
- `CatalysisRegime.C`, `CatalysisRegime.C0`: `C^\theta`, `C_0^\theta`
- `CatalysisRegime.PΠ`, `CatalysisRegime.CΠ`: x-space catalysis coefficients
- `BncRegime.H`: `(q_ss, K, k) -> x`
- `BncRegime.H0`: reduced affine offset after expanding `K_ss`
- `BncRegime.C_qKk_cat`, `BncRegime.C0_qKk_cat`: mixed consistency in `(q,K,k)`
- `BncRegime.C_qKk_ss`, `BncRegime.C0_qKk_ss`: steady-state consistency in `(q_ss,K,k)`
- `BncRegime.H_bd`: stability screening matrix
