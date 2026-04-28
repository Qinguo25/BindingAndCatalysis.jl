module DStable

export judge_dstable, d_class

using SparseArrays
using LinearAlgebra
import JuMP
import MathOptInterface as MOI
import Arpack
import Clarabel


@inline tri_index(i::Int, j::Int) = (j * (j - 1)) ÷ 2 + i  # 要求 i <= j

function _default_optimizer_factory(time_limit::Real)
    return JuMP.optimizer_with_attributes(
        Clarabel.Optimizer,
        "verbose" => false,
        "time_limit" => Float64(time_limit),
    )
end

function _build_model(optimizer_factory)
    model = JuMP.Model(optimizer_factory)
    try
        set_silent(model)
    catch
        # 某些优化器不支持 MOI.Silent()，忽略即可
    end
    return model
end

"""
    _obvious_not_hurwitz(A; ...)

单边测试：
- 返回 `true`     : 已证实 A 不是 Hurwitz
- 返回 `:unknown` : 无法确定
"""
function _obvious_not_hurwitz(
    A::SparseMatrixCSC{Float64, Int};
    dense_exact_threshold::Int = 256,
    spectral_tol::Float64 = 1e-8,
    eigs_tol::Float64 = 1e-8,
    eigs_maxiter::Int = 2000,
)
    n = size(A, 1)

    # 小规模：直接精确（数值上）算全部特征值
    if n <= dense_exact_threshold
        α = maximum(real.(eigvals(Matrix(A))))
        return α >= -spectral_tol
    end

    # 大规模：ARPACK 做单边筛查
    # 若 Ritz 对满足 Re(λ̂) - residual > 0，则可保守判定存在 RHP 特征值
    try
        λ, V, nconv, _, _, _ = Arpack.eigs(
            A;
            nev = 1,
            which = :LR,      # largest real part
            tol = eigs_tol,
            maxiter = eigs_maxiter,
            ritzvec = true,
        )
        if nconv >= 1
            λ1 = λ[1]
            v1 = V[:, 1]
            r  = norm(A * v1 - λ1 * v1) / max(norm(v1), eps(Float64))
            if real(λ1) - r > spectral_tol
                return true
            end
        end
    catch
        # 留给后续 SDP
    end

    return :unknown
end

"""
构造 -(A'P + P*A) - tI 的上三角向量化，
其中 P = diag(signs .* x).
如果 signs === nothing，则表示全正号。
"""
function _neg_lyap_triangle(
    A::SparseMatrixCSC{Float64, Int},
    x,
    t;
    signs::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = size(A, 1)
    tri = [JuMP.AffExpr(0.0) for _ in 1:(n * (n + 1) ÷ 2)]

    # 对角：-t I
    for i in 1:n
        JuMP.add_to_expression!(tri[tri_index(i, i)], -1.0, t)
    end

    rows = rowvals(A)
    vals = nonzeros(A)

    # 对每个非零 a_{row,col}：
    # 上三角位置 (min(row,col), max(row,col))
    # 接收 a_{row,col} * p_row 的贡献；对角项会自动翻倍
    for col in 1:n
        for ptr in nzrange(A, col)
            row = rows[ptr]
            a   = vals[ptr]

            i = min(row, col)
            j = max(row, col)
            idx = tri_index(i, j)

            s = isnothing(signs) ? 1.0 : Float64(signs[row])
            coeff = -(row == col ? 2.0 : 1.0) * a * s
            JuMP.add_to_expression!(tri[idx], coeff, x[row])
        end
    end

    return tri
end

"""
求解
    max t
    s.t. -(A'P + P*A) - tI ⪰ 0,
         P = diag(signs .* x),
         x_i >= p_floor,
         sum(x) = 1.
返回最优 t；若求解失败则返回 -Inf。
"""
function _signed_diag_lyap_margin(
    A::SparseMatrixCSC{Float64, Int};
    optimizer_factory,
    p_floor::Float64 = 1e-8,
    signs::Union{Nothing, AbstractVector{<:Real}} = nothing,
)
    n = size(A, 1)
    n * p_floor < 1 || throw(ArgumentError("需要满足 n * p_floor < 1"))

    model = _build_model(optimizer_factory)

    JuMP.@variable(model, x[1:n] >= p_floor)
    JuMP.@variable(model, t >= 0.0)
    JuMP.@constraint(model, sum(x) == 1.0)

    tri = _neg_lyap_triangle(A, x, t; signs = signs)
    JuMP.@constraint(model, tri in MOI.PositiveSemidefiniteConeTriangle(n))
    JuMP.@objective(model, Max, t)

    JuMP.optimize!(model)

    st = JuMP.termination_status(model)
    if st == MOI.OPTIMAL || st == MOI.ALMOST_OPTIMAL
        return JuMP.value(t)
    end
    return -Inf
end

"""
判据 B 的可选备份：
只枚举“恰好一个负号”的模式。
找到即可给出强 0 证书。
"""
function _strong_zero_certificate_singletons(
    A::SparseMatrixCSC{Float64, Int};
    optimizer_factory,
    p_floor::Float64 = 1e-8,
    margin_tol::Float64 = 1e-7,
    max_patterns::Int = 16,
)
    n = size(A, 1)
    n == 0 && return false
    max_patterns <= 0 && return false

    # 启发式：先试对角元较大的位置
    ord = sortperm(diag(A); rev = true)
    kmax = min(n, max_patterns)

    for kk in 1:kmax
        k = ord[kk]
        signs = ones(Float64, n)
        signs[k] = -1.0

        t = _signed_diag_lyap_margin(
            A;
            optimizer_factory = optimizer_factory,
            p_floor = p_floor,
            signs = signs,
        )
        if isfinite(t) && t > margin_tol
            return true
        end
    end

    return false
end







"""
    judge_dstable(A; kwargs...) -> Int

输出：
- 1  : 证实 D-stable（通过 diagonal stability）
- 0  : 证实一定不 D-stable
- -1 : 无法判断

默认是“性能优先”：
1) 先做单边的非 Hurwitz 筛查；
2) 再做 diagonal stability SDP；
3) 默认不跑判据 B 的符号枚举；
   若要启用，把 try_strong_zero=true。
"""
function judge_dstable(
    Ain::AbstractMatrix{<:Real};
    # optimizer_factory = nothing,
    time_limit::Float64 = 20.0,
    dense_exact_threshold::Int = 256,
    spectral_tol::Float64 = 1e-8,
    eigs_tol::Float64 = 1e-8,
    eigs_maxiter::Int = 2000,
    p_floor::Float64 = 1e-8,
    margin_tol::Float64 = 1e-7,
    try_strong_zero::Bool = false,
    strong_zero_patterns::Int = 16,
)::Int
    n, m = size(Ain)
    n == m || throw(ArgumentError("A 必须是方阵"))

    if n == 0
        return 1
    end

    A = sparse(Float64.(Ain))
    all(isfinite, nonzeros(A)) || throw(ArgumentError("A 含 NaN/Inf"))

    # 归一化，改善数值条件；不改变稳定性符号
    scale = max(opnorm(A, Inf), 1.0)
    A = A / scale

    # if optimizer_factory === nothing
        optimizer_factory = _default_optimizer_factory(time_limit)
    # end

    # Step 0: 便宜的单边 0 证书
    rhp = _obvious_not_hurwitz(
        A;
        dense_exact_threshold = dense_exact_threshold,
        spectral_tol = spectral_tol,
        eigs_tol = eigs_tol,
        eigs_maxiter = eigs_maxiter,
    )
    if rhp === true
        return 0
    end

    # Step 0.1: for lesser than 3 dim matrix, we know explicitly the d-stable conditon
    
    if n <= 3
        return d_class(Matrix(A); tol = margin_tol)
    end


    # Step 1: diagonal stability => D-stable
    tpos = _signed_diag_lyap_margin(
        A;
        optimizer_factory = optimizer_factory,
        p_floor = p_floor,
        signs = nothing,
    )
    if isfinite(tpos) && tpos > margin_tol
        return 1
    end

    # Step 2: 可选的强 0 证书备份（只在谱筛查未明确时有意义）
    if try_strong_zero && rhp === :unknown
        if _strong_zero_certificate_singletons(
            A;
            optimizer_factory = optimizer_factory,
            p_floor = p_floor,
            margin_tol = margin_tol,
            max_patterns = strong_zero_patterns,
        )
            return 0
        end
    end

    return -1
end





"""
    d_class(A; tol=1e-10)

For real square matrices A with size n = 1, 2, or 3:

Return:
    1   if A is D-stable
    0   if A is D-unstable
   -1   otherwise

Definition:
    D-stable:     D*A is Hurwitz for every positive diagonal D.
    D-unstable:   D*A is not Hurwitz for every positive diagonal D.
"""
function d_class(A::AbstractMatrix; tol::Real = 1e-10)
    n, m = size(A)
    n == m || throw(ArgumentError("A must be square."))
    1 <= n <= 3 || throw(ArgumentError("Only n = 1, 2, 3 are supported."))

    all(isreal, A) || throw(ArgumentError("A must be real."))

    B = Float64.(real.(A))

    pos(x) = x > tol
    neg(x) = x < -tol
    nonpos(x) = x <= tol
    nonneg(x) = x >= -tol

    # ---------- n = 1 ----------
    if n == 1
        a = B[1, 1]

        if neg(a)
            return 1          # always Hurwitz after positive scaling
        else
            return 0          # never Hurwitz
        end
    end

    # ---------- n = 2 ----------
    if n == 2
        a = B[1, 1]
        d = B[2, 2]
        detA = B[1,1] * B[2,2] - B[1,2] * B[2,1]

        # D-stable iff det(A)>0, a<=0, d<=0, and not both a,d are zero.
        if pos(detA) && nonpos(a) && nonpos(d) && neg(a + d)
            return 1
        end

        # D-unstable iff det(A)<=0 or both diagonal entries are nonnegative.
        if !pos(detA) || (nonneg(a) && nonneg(d))
            return 0
        end

        return -1
    end

    # ---------- n = 3 ----------
    a11, a12, a13 = B[1,1], B[1,2], B[1,3]
    a21, a22, a23 = B[2,1], B[2,2], B[2,3]
    a31, a32, a33 = B[3,1], B[3,2], B[3,3]

    # alpha_i = -a_ii
    alpha = [-a11, -a22, -a33]

    # Principal 2x2 minors
    Δ12 = a11 * a22 - a12 * a21
    Δ13 = a11 * a33 - a13 * a31
    Δ23 = a22 * a33 - a23 * a32

    # beta is ordered so that
    # p2/(xyz) = beta[1]/x + beta[2]/y + beta[3]/z
    beta = [Δ23, Δ13, Δ12]

    delta = -det(B)




    # ---------- Check D-stability for n = 3 ----------
    #
    # Need:
    #   alpha_i >= 0, not all zero
    #   beta_i  >= 0, not all zero
    #   delta > 0
    #   inf_x (alpha⋅x)(beta⋅1/x) > delta
    #
    # The infimum is:
    #   (sum_i sqrt(alpha_i * beta_i))^2
    #
    alpha_ok = all(nonneg, alpha) && any(pos, alpha)
    beta_ok  = all(nonneg, beta)  && any(pos, beta)

    if pos(delta) && alpha_ok && beta_ok
        lower = sum(sqrt(max(alpha[i], 0.0) * max(beta[i], 0.0)) for i in 1:3)^2

        # If supports differ, the infimum is not attained inside x>0.
        support_equal = all((alpha[i] > tol) == (beta[i] > tol) for i in 1:3)

        if delta < lower - tol || (abs(delta - lower) <= tol && !support_equal)
            return 1
        end
    end


    # ---------- Check D-stabilizability for n = 3 ----------
    #
    # A is D-unstable iff it is NOT possible to find a positive diagonal D
    # such that D*A is Hurwitz.
    #
    # For n=3 this reduces to checking whether
    #
    #   sup_{x>0, alpha⋅x>0} (alpha⋅x)(beta⋅1/x) > delta.
    #
    function is_d_stabilizable_3(alpha, beta, delta)
        pos(delta) || return false

        P = findall(i -> alpha[i] > tol, 1:3)

        # Need alpha⋅x > 0 for some positive x.
        isempty(P) && return false

        # Need beta⋅1/x positive somewhere.
        any(i -> beta[i] > tol, 1:3) || return false

        # If at least two alpha_i are positive and some beta_j is positive,
        # the supremum is +∞.
        if length(P) >= 2
            return true
        end

        # Exactly one positive alpha.
        k = P[1]

        # If some beta_j > 0 with j != k, again the supremum is +∞.
        for j in 1:3
            if j != k && beta[j] > tol
                return true
            end
        end

        # Now the only possible positive beta is beta[k].
        beta[k] > tol || return false

        C = alpha[k] * beta[k]

        S = 0.0
        for j in 1:3
            j == k && continue

            if alpha[j] < -tol && beta[j] < -tol
                S += sqrt((-alpha[j]) * (-beta[j]) / C)
            end
        end

        M = C * max(1.0 - S, 0.0)^2

        return M > delta + tol
    end


    if is_d_stabilizable_3(alpha, beta, delta)
        return -1
    else
        return 0
    end

end

end # module DStable
