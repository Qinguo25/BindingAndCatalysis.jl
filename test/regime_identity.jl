@testset "Immutable Regime And Path Identities" begin
    model = minimal_catalysis_model()
    match_regimes!(model; warn_singular_propagation=false)

    binding_perms = get_binding_perms(model)
    binding_perm = copy(first(binding_perms))
    binding_key = Tuple(binding_perm)
    binding_dict = get_binding_regimes_dict(model)
    binding_idx = binding_dict[binding_key]

    @test keytype(binding_dict) <: Tuple
    @test all(key -> key isa Tuple, keys(binding_dict))
    @test get_binding_index(model, binding_key; check=true) == binding_idx
    @test get_binding_regime(model, binding_key) === get_binding_regime(model, binding_idx)
    @test get_binding_perm(model, binding_key; check=true) == binding_perm
    @test have_perm(model, binding_key)

    numeric_symbols = x_symbol(model)[binding_perm]
    plain_symbols = Symbol.(string.(numeric_symbols))
    for selector in
        (numeric_symbols, plain_symbols, Tuple(numeric_symbols), Tuple(plain_symbols))
        @test get_binding_perm(model, selector; check=true) == binding_perm
        @test get_binding_index(model, selector; check=true) == binding_idx
        @test get_binding_regime(model, selector) === get_binding_regime(model, binding_idx)
    end

    stored_perm = get_binding_regime(model, binding_idx).perm
    original_perm = copy(stored_perm)
    try
        stored_perm[1] += 10_000
        @test binding_dict[Tuple(original_perm)] == binding_idx
    finally
        copyto!(stored_perm, original_perm)
    end

    catalysis = get_catalysis_network(model)
    catalysis_dict = get_catalysis_regimes_dict(catalysis)
    catalysis_key = first(keys(catalysis_dict))
    catalysis_idx = catalysis_dict[catalysis_key]

    @test keytype(catalysis_dict) <: Tuple
    @test get_catalysis_index(catalysis, catalysis_key; check=true) == catalysis_idx
    @test get_catalysis_perm(catalysis, catalysis_key; check=true) == collect(catalysis_key)
    @test get_catalysis_regime(catalysis, catalysis_key) ===
        get_catalysis_regime(catalysis, catalysis_idx)
    @test have_perm(catalysis, catalysis_key)

    paths = SIMOPaths(minimal_model(), 1)
    path_dict = BindingAndCatalysis._ensure_paths_dict!(paths)
    path = paths.rgm_paths[1]
    path_copy = copy(path)
    path_key = Tuple(path_copy)

    @test keytype(path_dict) <: Tuple
    @test all(key -> key isa Tuple, keys(path_dict))
    @test get_idx(paths, path_copy) == path_dict[path_key]
    @test get_idx(paths, path_key) == path_dict[path_key]
    @test get_path(paths, path_key; return_idx=true) == path_copy

    try
        path[1] += 10_000
        @test path_dict[path_key] == 1
    finally
        copyto!(path, path_copy)
    end
end

@testset "Catalysis Balance Equality Semantics" begin
    model = minimal_catalysis_model()
    catalysis_regimes = get_catalysis_regimes(model)
    regime = first(catalysis_regimes)

    @test balance_equality_count(regime) == get_catalysis_network(model).r_v
    @test get_catalysis_regimes(model; singular=false) == catalysis_regimes
    @test isempty(get_catalysis_regimes(model; singular=true))

    error_value = try
        get_nullity(regime)
        nothing
    catch err
        err
    end
    @test error_value isa ArgumentError
    @test occursin("balance_equality_count", sprint(showerror, error_value))

    shown = sprint(show, MIME("text/plain"), regime)
    @test occursin("balance equality count", shown)
    @test !occursin("nullity", shown)
end

@testset "BNC Assignment Uses Global Regime Identity" begin
    model = minimal_catalysis_model()
    match_regimes!(model; warn_singular_propagation=false)
    all_regimes = get_bnc_regimes(model; feasible=nothing)
    @test length(all_regimes) >= 2

    first_regime = all_regimes[1]
    original_feasibility = first_regime.is_feasible
    try
        first_regime.is_feasible = false
        candidates = get_bnc_regimes(model)
        expected = first(candidates)
        logwKk = zeros(length(wKk_symbol(model)))
        C, C0, nullity = get_C_C0_nullity_wKk(expected)

        @test get_idx(expected) > 1
        @test BindingAndCatalysis.condition_contains(C, C0, nullity, logwKk)
        @test assign_bnc_regime_wKk(model, logwKk) == get_idx(expected)
    finally
        first_regime.is_feasible = original_feasibility
    end
end

@testset "Reaction-Order NaN Segment Deduplication" begin
    B = BindingAndCatalysis

    @test isempty(B._dedup(Float64[]))
    @test isequal(B._dedup([NaN]), [NaN])
    @test isequal(
        B._dedup([NaN, NaN, 1.0, 1.0, NaN, NaN, 1.0, 2.0, 2.0, NaN]),
        [NaN, 1.0, NaN, 1.0, 2.0, NaN],
    )
end
