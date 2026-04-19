"@stdlib" ∉ LOAD_PATH && push!(LOAD_PATH, "@stdlib")

import Pkg
import Pkg.Artifacts
using Pkg.Artifacts: ensure_artifact_installed, artifact_hash, artifact_path

root = normpath(joinpath(@__DIR__, ".."))
script = joinpath(root, "scripts", "build_local_cdd.sh")
bindir = joinpath(root, ".build", "cddlog", "src")
artifacts_toml = joinpath(root, "Artifacts.toml")
artifact_name = "cddlog_source"
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

function _normalize_cddlog_source_root(root::AbstractString)
    isdir(root) || return nothing
    if isfile(joinpath(root, "lib-src", "cddcore.c"))
        return root
    end
    subdirs = filter(name -> isdir(joinpath(root, name)), readdir(root))
    if length(subdirs) == 1
        nested = joinpath(root, only(subdirs))
        if isfile(joinpath(nested, "lib-src", "cddcore.c"))
            return nested
        end
    end
    return nothing
end

function _resolve_cddlog_source_root()
    if haskey(ENV, "BNC_CDDLOG_SOURCE_DIR")
        requested = ENV["BNC_CDDLOG_SOURCE_DIR"]
        src_root = _normalize_cddlog_source_root(requested)
        src_root === nothing && error("BNC_CDDLOG_SOURCE_DIR does not look like a cddlib-logarithmic source tree: $requested")
        return src_root
    end

    isfile(artifacts_toml) || error("Artifacts.toml is missing at $artifacts_toml")
    ensure_artifact_installed(artifact_name, artifacts_toml; quiet_download=false)
    hash = artifact_hash(artifact_name, artifacts_toml)
    hash === nothing && error("Artifact '$artifact_name' is not defined in $artifacts_toml")
    src_root = _normalize_cddlog_source_root(artifact_path(hash))
    src_root === nothing && error("Artifact '$artifact_name' was installed, but no cddlib-logarithmic source tree was found below $(artifact_path(hash))")
    return src_root
end

if !isfile(script)
    @warn "Local cdd build script is missing; skipping build." script
elseif _have_required_tools(bindir, required) && get(ENV, "BNC_FORCE_REBUILD_CDD", "0") != "1"
    @info "Local cdd backend already available." bindir
else
    try
        src_root = _resolve_cddlog_source_root()
        run(addenv(`bash $script`, "BNC_CDDLOG_SOURCE_DIR" => src_root))
        _have_required_tools(bindir, required) || error("Local cdd build finished but expected tools are missing in $bindir")
        @info "Local cdd backend build completed." bindir src_root
    catch err
        @error "Local cdd backend build failed. Install gcc/cc/clang and libgmp-dev, then rerun `Pkg.build()`." exception=(err, catch_backtrace())
        rethrow(err)
    end
end
