@testset "Top-row perturbation indices" begin
    A = sparse([
        0.0 1.0 0.0
        1.0 0.0 0.0
        0.0 0.0 1.0
    ])

    idxs = BindingAndCatalysis.diag_indices(A, 2)
    @test length(idxs) == 2

    kept = copy(A)
    keep_mask = falses(length(kept.nzval))
    keep_mask[idxs] .= true
    for col in axes(kept, 2)
        for nzidx in kept.colptr[col]:(kept.colptr[col + 1] - 1)
            if kept.rowval[nzidx] <= 2 && !keep_mask[nzidx]
                kept.nzval[nzidx] = 0
            end
        end
    end
    dropzeros!(kept)

    @test rank(Matrix(kept)) == 3
    @test sort([A.rowval[idx] => findfirst(>(idx), A.colptr) - 1 for idx in idxs]) == [1 => 2, 2 => 1]
end
