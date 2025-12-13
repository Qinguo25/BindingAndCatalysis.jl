#-----------------------------------------------------------------
# Functions for assigning vertices
#-----------------------------------------------------------------


function assign_vertex_x(Bnc::Bnc{T}, x::AbstractVector{<:Real};input_logspace::Bool=false,asymptotic::Bool=true)::Vector{T} where T
    # x = input_logspace ? exp10.(x) : x
    L = Bnc._L_sparse
    d = Bnc.d
    n = Bnc.n
    max_indices = zeros(T, d)
    max_val = fill(-Inf, d)
    colptr = L.colptr
    rowval = L.rowval

    if asymptotic
        nzval = @view(x[Bnc._LN_top_cols])
    else
        x = input_logspace ? exp10.(x) : x # linear or log space only matters when not asymptotic
        nzval = @view(x[Bnc._LN_top_cols]) .* L.nzval
    end

    @inbounds for col in 1:n
        col_start_idx = colptr[col]
        col_end_idx   = colptr[col+1] - 1
        if col_start_idx <= col_end_idx #escape empty column
            @inbounds for idx in col_start_idx:col_end_idx
                v = nzval[idx]
                row = rowval[idx]
                if v > max_val[row]
                    max_val[row] = v
                    max_indices[row] = col
                end
            end
        end
    end
    return max_indices
end


# function get_vertex_qK(Bnc::Bnc, x::AbstractMatrix{<:Real}; kwargs...) 
#     [get_vertex_qK_slow(Bnc, row; kwargs...) for row in eachrow(x)]
# end

function assign_vertex_qK(Bnc::Bnc ; x::AbstractVector{<:Real}, input_logspace::Bool=false, kwargs...) 
    # @show all_vertice_idx
    logqK = x2qK(Bnc,x; input_logspace=input_logspace, output_logspace=true)
    return assign_vertex_qK(Bnc, logqK; input_logspace=true, kwargs...)
end


function assign_vertex_qK(Bnc::Bnc, qK::AbstractVector{<:Real}; input_logspace::Bool=false, asymptotic::Bool=false, eps=0) 
    real_only = asymptotic ? true : nothing
    all_vertice_idx = get_vertices(Bnc, singular=false, asymptotic = real_only, return_idx = false)
    # @show all_vertice_idx
    logqK = input_logspace ? qK : log10.(qK)
    
    record = Vector{Float64}(undef,length(all_vertice_idx))
    for (i, idx) in enumerate(all_vertice_idx)
        C, C0 = get_C_C0_qK!(Bnc, idx) 
        
        min_val = if !asymptotic
            minimum(C * logqK .+ C0)
        else
            minimum(C * logqK)
        end
        # @show idx
        # @show min_val
        record[i] = min_val

        if record[i] >= -eps
            return idx
        end
    end
    @warn("All vertex conditions failed for logqK=$logqK. Returning the best-fit vertex.")
    return all_vertice_idx[findmax(record)[2]]
end
#-------------------------------------------------------------------------------------------------------------------------------------------------------

# Trying speedup assign_vertex_qK, but not success yet.
function get_i_j(model::Bnc,perm::Vector{<:Integer}, t::Integer)
    i = findfirst(>(t),model._C_partition_idx) - 1
    j1 = perm[i]
    cth = t - model._C_partition_idx[i] + 1
    j2 = model._valid_L_idx[i][cth]
    j2 < j1 ? nothing : j2 += 1
    return i, j1, j2
end

function assign_vertex_qK_test(Bnc::Bnc{T}, qK::AbstractVector{<:Real};
                               input_logspace::Bool=false,
                               asymptotic::Bool=true, eps=0) where T
    logqK = input_logspace ? qK : log10.(qK)
    Perm_tried = Set{UInt64}()  # 存放哈希值

    function try_perm!(perm1)
        (C, C0) = get_C0_qK!(Bnc, perm1)
        err = C * logqK .+ C0
        ts = findall(er -> er <= -eps, err)

        # 没有违反不等式，返回
        if isempty(ts)
            return perm1
        end

        h = hash(perm1)
        if h in Perm_tried
            error("Cyclic permutation detected! Tried permutations: $(collect(Perm_tried))")
        end
        push!(Perm_tried, h)

        # 对所有违反的约束更新 perm
        for t in ts
            i, j1, j2 = get_i_j(Bnc, perm1, t)
            perm1[i] = j2
            if !haskey(Bnc.vertices_perm_dict, perm1) 
                perm1[i] = j1  # 恢复原值
            end
            try_perm!(perm1)
        end
        # else
        #         @show perm1
    end

    # 假设初始 perm1 为 1:Bnc.d 或者外部传入
    perm0 = collect(1:Bnc.d) .|> x->T(x)
    return try_perm!(perm0)
end