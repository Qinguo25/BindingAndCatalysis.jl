@testset "Standard Sparse Inverse Solves" begin
    B = BindingAndCatalysis
    A = sparse([
        4.0 1.0 0.0
        2.0 3.0 1.0
        0.0 1.0 2.0
    ])
    identity_dense = Matrix{Float64}(I, size(A)...)

    entry = B._factor_Nρ(A; drop_tol=0.0)
    @test entry.kind == 0x01
    @test entry.inv isa SparseMatrixCSC
    @test Matrix(A * entry.inv) ≈ identity_dense

    H, nullity = B.direct_inverse_or_adjugate(A)
    @test nullity == 0
    @test H isa SparseMatrixCSC
    @test Matrix(A * H) ≈ identity_dense

    rhs = spdiagm(0 => ones(Float64, size(A, 1)))
    sparse_ldiv_method = which(getfield(Base, Symbol("\\")), (typeof(A), typeof(rhs)))
    sparse_inv_method = which(inv, (typeof(A),))
    @test sparse_ldiv_method.module !== B
    @test sparse_inv_method.module !== B
    @test !isdefined(B, :luFac)
end
