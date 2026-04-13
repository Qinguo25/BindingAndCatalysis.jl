using BindingAndCatalysis
using LinearAlgebra
using Random
using SparseArrays
using Test

function minimal_model()
    N = [1 1 -1]
    x_sym = [:E, :S, :C]
    q_sym = [:tE, :tS]
    K_sym = [:K]
    return Bnc(N = N, x_sym = x_sym, q_sym = q_sym, K_sym = K_sym)
end

function sparse_singular_model()
    L = sparse(
        [1, 2, 3, 4, 4, 1, 3, 4, 2, 4],
        [1, 2, 3, 3, 4, 5, 5, 5, 6, 6],
        ones(Int, 10),
        4,
        6,
    )
    N = sparse(
        [1, 2, 1, 2, 1, 2],
        [1, 2, 3, 4, 5, 6],
        [1, 1, 1, 1, -1, -1],
        2,
        6,
    )
    return Bnc(L = L, N = N)
end

function minimal_catalysis_model()
    model = minimal_model()
    update_catalysis!(
        model;
        Γ = [1 -1],
        Π = [1 0 0; 0 1 0],
        q_picked = [:tE],
        k_sym = [:k1, :k2],
    )
    return model
end

function offset_catalysis_model()
    model = minimal_model()
    update_catalysis!(
        model;
        Γ = [2 1 -1],
        Π = [1 0 0; 0 1 0; 0 0 1],
        q_picked = [:tE],
        k_sym = [:k1, :k2, :k3],
    )
    return model
end

function notebook_model2()
    N = [
        1 1 -1 0 0
        1 0 1 -1 0
        0 1 0 1 -1
    ]
    return Bnc(N = N)
end

function clique5_binding_model()
    N = [
        1 1 0 0 0 -1 0 0 0 0 0 0 0 0 0
        1 0 1 0 0 0 -1 0 0 0 0 0 0 0 0
        1 0 0 1 0 0 0 -1 0 0 0 0 0 0 0
        1 0 0 0 1 0 0 0 -1 0 0 0 0 0 0
        0 1 1 0 0 0 0 0 0 -1 0 0 0 0 0
        0 1 0 1 0 0 0 0 0 0 -1 0 0 0 0
        0 1 0 0 1 0 0 0 0 0 0 -1 0 0 0
        0 0 1 1 0 0 0 0 0 0 0 0 -1 0 0
        0 0 1 0 1 0 0 0 0 0 0 0 0 -1 0
        0 0 0 1 1 0 0 0 0 0 0 0 0 0 -1
    ]
    x_sym = [:A, :B, :C, :D, :E, :ab, :ac, :ad, :ae, :bc, :bd, :be, :cd, :ce, :de]
    q_sym = [:tA, :tB, :tC, :tD, :tE]
    K_sym = [:K12, :K13, :K14, :K15, :K23, :K24, :K25, :K34, :K35, :K45]
    return Bnc(N = N, x_sym = x_sym, q_sym = q_sym, K_sym = K_sym)
end
