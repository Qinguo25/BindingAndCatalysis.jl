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
    binding_snapshots = (
        get_binding_perm(model, binding_idx),
        get_binding_perm(get_binding_regime(model, binding_idx)),
        get_perm(model, binding_idx),
        get_perm(get_binding_regime(model, binding_idx)),
        get_binding_perms(model)[binding_idx],
    )
    for snapshot in binding_snapshots
        @test snapshot == original_perm
        @test snapshot !== stored_perm
        snapshot[1] += 10_000
        @test stored_perm == original_perm
        @test binding_dict[Tuple(original_perm)] == binding_idx
    end

    neighbor_perms = get_neighbors(model, binding_idx)
    @test !isempty(neighbor_perms)
    neighbor_idx = get_binding_index(model, first(neighbor_perms))
    stored_neighbor_perm = get_binding_regime(model, neighbor_idx).perm
    @test first(neighbor_perms) !== stored_neighbor_perm
    neighbor_original = copy(stored_neighbor_perm)
    first(neighbor_perms)[1] += 10_000
    @test stored_neighbor_perm == neighbor_original

    assigned_idx = assign_regime_x(
        model, ones(model.n); asymptotic_only=false, return_idx=true
    )
    assigned_perm = assign_regime_x(model, ones(model.n); asymptotic_only=false)
    stored_assigned_perm = get_binding_regime(model, assigned_idx).perm
    assigned_original = copy(stored_assigned_perm)
    @test assigned_perm == assigned_original
    @test assigned_perm !== stored_assigned_perm
    assigned_perm[1] += 10_000
    @test get_binding_regime(model, assigned_idx).perm == assigned_original

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

    stored_catalysis_perm = get_catalysis_regime(catalysis, catalysis_idx).perm
    catalysis_original = copy(stored_catalysis_perm)
    catalysis_snapshots = (
        get_catalysis_perm(catalysis, catalysis_idx),
        get_catalysis_perm(get_catalysis_regime(catalysis, catalysis_idx)),
        get_perm(catalysis, catalysis_idx),
        get_perm(get_catalysis_regime(catalysis, catalysis_idx)),
        get_catalysis_perms(catalysis)[catalysis_idx],
    )
    for snapshot in catalysis_snapshots
        @test snapshot == catalysis_original
        @test snapshot !== stored_catalysis_perm
        snapshot[1] += 10_000
        @test stored_catalysis_perm == catalysis_original
        @test catalysis_dict[Tuple(catalysis_original)] == catalysis_idx
    end

    bnc_regime = first(get_bnc_regimes(model))
    binding_snapshot, catalysis_snapshot = get_perm(bnc_regime)
    @test binding_snapshot !== bnc_regime.bind_rgm.perm
    @test catalysis_snapshot !== bnc_regime.catalysis_rgm.perm
    collection_binding_snapshot, collection_catalysis_snapshot = first(get_bnc_perms(model))
    @test collection_binding_snapshot !== bnc_regime.bind_rgm.perm
    @test collection_catalysis_snapshot !== bnc_regime.catalysis_rgm.perm
    binding_snapshot[1] += 10_000
    catalysis_snapshot[1] += 10_000
    @test bnc_regime.bind_rgm.perm == get_binding_perm(model, bnc_regime.bind_rgm.idx)
    @test bnc_regime.catalysis_rgm.perm ==
        get_catalysis_perm(catalysis, bnc_regime.catalysis_rgm.idx)

    _, steady_catalysis_snapshot = get_steady_state_perm(bnc_regime)
    @test steady_catalysis_snapshot !== bnc_regime.catalysis_rgm.perm

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

    returned_path = get_path(paths, 1; return_idx=true)
    @test returned_path == path_copy
    @test returned_path !== path
    returned_path[1] += 10_000
    @test path == path_copy
    @test path_dict[path_key] == 1

    returned_perm_path = get_path(paths, 1)
    for (snapshot, regime_idx) in zip(returned_perm_path, path)
        stored_path_perm = get_binding_regime(paths.bn, regime_idx).perm
        @test snapshot == stored_path_perm
        @test snapshot !== stored_path_perm
        original = copy(stored_path_perm)
        snapshot[1] += 10_000
        @test stored_path_perm == original
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
