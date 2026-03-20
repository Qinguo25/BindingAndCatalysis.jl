
function get_binding_network(model::CatalysisData)
        if isnothing(model.network)
            @warn "Binding Network not found in the model"
            return nothing
        end
    return model.network
end


function get_catalysis_network(model::Bnc)
        if isnothing(model.catalysis)
            @warn "Catalysis Network not found in the model"
            return nothing
        end
    return model.catalysis
end
get_catalysis_network(model::CatalysisData) = model




function find_catalysis_regimes!(model::Bnc)
    find_catalysis_regimes!(get_catalysis_network(model))
end



function find_catalysis_regimes!(model::CatalysisData;)
    if !isnothing(model.CatalysisRegimes)
        return nothing
    end

    @info "---------------------Start finding all vertices--------------------"
    all_vertices, is_asymptotic =  _enumerate_all_regimes(model._S_helper)
    # all_vertices = [Vector{T}(v) for v in all_vertices]

    n_vertices = length(all_vertices)
    n_asym_rgms = sum(is_asymptotic)
    @info "Finished, with $(n_vertices) catalysis vertices found and $(n_asym_rgms) asymptotic vertices."
    
    @info "3.Building Regimes..."
    model.CatalysisRegimes = let
        regimes = _build_catalysis_regimes(model, all_vertices, is_asymptotic)    
        vertices_perm_dict = Dict(perm => idx for (idx, perm) in enumerate(all_vertices))
        Regimes(vertices_perm_dict, regimes)
    end
    return nothing
end

@inline function _build_catalysis_regimes(model::CatalysisData, all_vertices, is_asymptotic)
    n_vertices = length(all_vertices)
    regimes = Vector{CatalysisRegime}(undef, n_vertices)
    for i in 1:n_vertices
        regimes[i] = CatalysisRegime(
            network = model, # network
            perm = all_vertices[i], #perm
            idx = i, # idx
            is_asymptotic = is_asymptotic[i]
        )
    end
    return regimes
end



function _initialize_regime!(vtx::CatalysisRegime)
    if !isnothing(vtx.P_pos_neg)
        return vtx
    end

    model = vtx.network
    perm = vtx.perm

    P_pos_neg, _ = _calc_P_P0(perm, model._S_helper)
    C, _ = _calc_C_C0(perm, model._S_helper)
    P = P_pos_neg[1:model.r_v, :] - P_pos_neg[model.r_v+1:end, :]

    # @show C, model.Π
    CΠ = C * model.Π
    PΠ = P * model.Π

    vtx.P_pos_neg = P_pos_neg
    vtx.P = P
    vtx.C = C
    vtx.CΠ = CΠ
    vtx.PΠ = PΠ
    return vtx
end

get_catalysis_regimes_dict(model::CatalysisData) = let
    find_catalysis_regimes!(model)
    get_regimes_dict(model.CatalysisRegimes)
end








function get_catalysis_regime(model::AbstractBnc, perm::AbstractVector{<:Integer};)
    cn = get_catalysis_network(model)
    return get_catalysis_regimes_dict(cn)[perm]
end


function get_C_k(rgm::CatalysisRegime)
    _initialize_regime!(rgm)
    return rgm.C
end
get_C_x(rgm::CatalysisRegime) = get_CΠ(rgm)
function get_CΠ(rgm::CatalysisRegime)
    _initialize_regime!(rgm)
    return rgm.CΠ
end

function get_C_xk(rgm::CatalysisRegime)
    _initialize_regime!(rgm)
    return hcat(rgm.CΠ, rgm.C)
end

function get_PΠ(rgm::CatalysisRegime)
    _initialize_regime!(rgm)
    return rgm.PΠ
end

function get_P(rgm::CatalysisRegime)
    _initialize_regime!(rgm)
    return rgm.P
end




# function _initialize_regime!(rgms::Matrix{Union{BncRegime,Nothing}})
#     first_pos = findfirst(x -> !isnothing(x), rgms)
#     first_pos === nothing && return nothing
#     first_vtx = rgms[first_pos]::BncRegime
#     r_v = size(first_vtx.catalysis_rgm.P, 1)
#     n = first_vtx.bind_rgm.network.n
    
#     @inline _calc_H2(H, P) = let
#         Hleft = H[:, 1:(n-r_v)]
#         Hright = H[:,n-r_v+1:end]
#         hcat(Hleft, Hright * (-P))
#     end

#     @info "Initializing BncRegimes, calculating H, H0, C_qKk_ss, C0_qKk_ss, C_qKk_cat, C0_qKk_cat for each regime..."
#     for i in 1:size(rgms, 1) # the row of Regimes share the same catalysis regime, so they share the same N_ss
        
#         N_ss = vcat(rgms[i, 1].bind_rgm.network.N, rgms[i, 1].catalysis_rgm.PΠ)
#         L_ss = rgms[i, 1].bind_rgm.network.L[r_v+1:end, :]

#         direction = sign(det(Matrix{Float64}(vcat(L_ss, N_ss))))

#         valid_js = [j for j in axes(rgms, 2) if !isnothing(rgms[i, j])]


#         perms = [get_perm(rgms[i, j]) for j in valid_js]

#         nlt, cache =  _calc_nullity(perms, N_ss)

#         Threads.@threads for k in eachindex(valid_js)
#             j = valid_js[k]
#             perm = perms[j]
#             vtx = rgms[i, j]::BncRegime
#             vtx.nlt = nlt[j]
#             if nlt[j] <=1
#                 Ptheta = vtx.catalysis_rgm.P
#                 P0_ss = @view vtx.bind_rgm.P0[r_v+1:end]
#                 M0_ss = vcat(P0_ss, zeros(eltype(P0_ss), size(N_ss, 1)))
#                 C_qKk_cat, C0_qKk_cat,C_qKk_cat_nlt = _calc_C_qKk_cat(vtx.bind_rgm, vtx.catalysis_rgm)
#                 vtx.C_qKk_cat = C_qKk_cat
#                 vtx.C0_qKk_cat = C0_qKk_cat
#                 vtx.nlt_qKk_cat = C_qKk_cat_nlt
                
#                 if nlt[j] == 0   
#                     H = _calc_H(N_ss, cache, perm)
#                     H2 = _calc_H2(H, Ptheta)
#                     H0_ss = -H * M0_ss
#                     vtx.H0 = H0_ss

#                     # Calculate C_qKk_ss and C0_qKk_ss, H, H0
#                     C_x_bind,C0_x_bind = get_C_C0_x(vtx.bind_rgm)
#                     # C0_qK_bind = C0_x_bind + C_x_bind * H0_ss
#                     C_x_cat = get_CΠ(vtx.catalysis_rgm)

#                     C_qKk_ss_upper = C_x_bind *H2
#                     C_qKk_ss_lower = let 
#                         d = size(Ptheta,2)
#                         C1 = C_x_cat * H2 
#                         n = size(C1, 2)
#                         C1[:,n-d+1:end] += vtx.catalysis_rgm.C
#                         C1
#                     end
#                     C_qKk_ss = vcat(C_qKk_ss_upper, C_qKk_ss_lower)

#                     C0_bind = C0_x_bind + C_x_bind * H0_ss
#                     C0_cat  = CΠ * H0_ss
#                     C0_qKk_ss = vcat(C0_bind, C0_cat)

#                     vtx.C_qKk_ss = C_qKk_ss
#                     vtx.C0_qKk_ss = C0_qKk_ss
#                     vtx.H = H2
#                     # Calculate C_qKk_cat and C0_qKk_ca
#                 else # nlt[j] == 1
#                     M = vcat(vtx.bind_rgm.P[r_v+1:end, :], N_ss)

#                     H = if length(Set(perm)) == length(perm)
#                             _calc_H(N_ss, cache, perm; scale = direction)
#                         else 
#                             H = _adj_singular_matrix(M)[1]
#                             H = droptol!(sparse(H),1e-10).* direction
#                             H
#                         end
#                     H2 = _calc_H2(H, Ptheta)
#                     vtx.H0 = -H * M0_ss
                    
#                     # Calculate C_qKk_ss and C0_qKk_ss, H, H0
#                     C_qKk_ss, C0_qKk_ss = let
#                         C_x_bind,C0_x_bind = get_C_C0_x(vtx.bind_rgm)    
#                         C_x_cat = get_CΠ(vtx.catalysis_rgm)

#                         C = hcat(C_x_bind,C_x_cat)
#                         C0 = vcat(C0_x_bind,zeros(size(C_x_cat, 1)))
#                         CqK,C0qK = _calc_C_C0_qK_singular(C,C0,M,M0_ss)

#                         d1 = size(C_x_bind,1)
#                         d2 = size(C_x_cat,1)

#                         C_qKk_ss_upper = CqK[:, 1:d1]
#                         C_qKk_ss_lower = let 
#                             d = size(Ptheta,2)
#                             C1 = CqK[:, d1+1:d1+d2] 
#                             n = size(C1, 2)
#                             C1[:,n-d+1:end] += vtx.catalysis_rgm.C
#                             C1
#                         end
#                         C_qKk_ss = vcat(C_qKk_ss_upper, C_qKk_ss_lower)

#                         C0_qKk_ss = C0qK
#                         (C_qKk_ss, C0_qKk_ss)
#                     end

#                     vtx.C_qKk_ss = C_qKk_ss
#                     vtx.C0_qKk_ss = C0_qKk_ss
#                     vtx.H = H2
#                     vtx.H0 = H0

#                     # Calculate C_qKk_cat and C0_qKk_cat

#                 end
#             end
#         end
#     end
#     @info "Finished initializing BncRegimes."
#     return nothing
# end





# @inline _calc_C_qKk_cat(bind_rgm::BindRegime, cat_rgm::CatalysisRegime) = let
#     if is_singular(bind_rgm)
#         C_x, C0 = get_C_C0_x(bind_rgm)
#         CΠ, Ctheta = get_C_xk(cat_rgm)
#         M,M0 = get_M_M0(bind_rgm)

#         n = size(M,1)
#         nv = size(Ctheta,2)
#         d1 = size(C_x,1)
#         d2 = size(CΠ,1)
        
#         C = let 
#             C1 = hcat(-diagm(n), spzeros(n,nv), M)
#             C2 = hcat(spzeros(d1,(n+nv)), C_x)
#             C3 = hcat(spzeros(d2,n),Ctheta, CΠ)
#             vcat(C1, C2, C3)
#         end

#         C0 = vcat(M0, C0, zeros(d2))

#         p = get_polyhedron(C,C0,n)
#         delset = BitSet((n+nv+1):(n+nv+n))
#         p2 = eliminate(p, delset)
#         return get_C_C0_nullity(p2)
#     else 
#         H,H0 = get_H_H0(bind_rgm)
#         C_qk,C0_qk = get_C_C0_qK(bind_rgm)
#         CΠ, Ctheta = get_C_xk(cat_rgm)
#         n_v = size(Ctheta, 2)
#         a = CΠ*H
#         C = let 
#             C1 = hcat(C_qk, spzeros(size(C_qk,1), n_v))
#             C2 = hcat(a,Ctheta)
#             vcat(C1, C2)
#         end
#         C0 = vcat(C0_qk, a * H0)
#         return C, C0, 0
#     end
# end




# function _build_BncRegime(cat_rgms, bind_rgms)
#     n_cat_rgms = length(cat_rgms.vertices_data)
#     n_bind_rgms = length(bind_rgms.vertices_data)
#     bncrgms = Matrix{Union{BncRegime,Nothing}}(undef, n_cat_rgms, n_bind_rgms)
#     @info "Matching Catalysis Regimes and Binding Regimes to build BncRegimes..."
#     Threads.@threads for i in 1:n_cat_rgms
#         for j in 1:n_bind_rgms
#             bncrgms[i, j] =if get_nullity(bind_rgms.vertices_data[j]) > 1
#                     nothing
#                 else
#                     BncRegime(
#                         bind_rgms.vertices_data[j],
#                         cat_rgms.vertices_data[i],
#                     )
#                 end
#         end
#     end
#     @info "Finished matching BncRegimes."
#     return bncrgms
# end

# function match_regimes!(model::Bnc)
#     find_all_regimes!(model)
#     find_catalysis_regimes!(model)
#     model.BncRegimes = _build_BncRegime(model.catalysis.CatalysisRegimes, model.BindRegimes)
#     _initialize_regime!(model.BncRegimes)
#     return nothing
# end


