root = normpath(joinpath(@__DIR__, ".."))
script = joinpath(root, "scripts", "build_local_cdd.sh")
bindir = joinpath(root, ".build", "cddlog", "src")
required = [
    "projection",
    "redcheck",
    "scdd",
    "projection_log",
    "redcheck_log",
    "scdd_log",
]

function _have_required_tools(dir, names)
    return isdir(dir) && all(name -> isfile(joinpath(dir, name)), names)
end

if !isfile(script)
    @warn "Local cdd build script is missing; skipping build." script
elseif _have_required_tools(bindir, required) && get(ENV, "BNC_FORCE_REBUILD_CDD", "0") != "1"
    @info "Local cdd backend already available." bindir
else
    try
        run(`bash $script`)
        _have_required_tools(bindir, required) || error("Local cdd build finished but expected tools are missing in $bindir")
        @info "Local cdd backend build completed." bindir
    catch err
        @warn "Local cdd backend build failed; runtime will fall back to NativePolyhedra where needed. Install gcc/cc/clang and libgmp-dev, then rerun `Pkg.build()`." exception=(err, catch_backtrace())
    end
end
