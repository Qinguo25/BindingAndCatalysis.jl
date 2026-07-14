__precompile__(false)

module BindingAndCatalysisVisualizationExt

using BindingAndCatalysis: BindingAndCatalysis
using GraphMakie: GraphMakie
using ImageFiltering: ImageFiltering
using Latexify: Latexify
using Makie: Makie

const _SRC_DIR = joinpath(dirname(@__DIR__), "src")
const _VIS_DIR = joinpath(_SRC_DIR, "visualization")

function _include_visualization(file::AbstractString)
    return Base.include(BindingAndCatalysis, joinpath(_VIS_DIR, file))
end

# The existing visualization source files are written inside the parent module.
# Bind the already-imported extension dependencies into that module without
# asking the parent package to import weak dependencies itself. Julia 1.12
# correctly rejects the latter even while the extension is active.
function _bind_optional_exports!(source::Module)::Nothing
    for name in names(source)
        Base.isidentifier(String(name)) || continue
        isdefined(BindingAndCatalysis, name) && continue
        value = getfield(source, name)
        Core.eval(BindingAndCatalysis, :(const $name = $value))
    end
    return nothing
end

for module_binding in (Makie, GraphMakie)
    name = nameof(module_binding)
    isdefined(BindingAndCatalysis, name) ||
        Core.eval(BindingAndCatalysis, :(const $name = $module_binding))
end
_bind_optional_exports!(Makie)
_bind_optional_exports!(GraphMakie)
_bind_optional_exports!(GraphMakie.NetworkLayout)
Core.eval(BindingAndCatalysis, :(const imfilter = $(ImageFiltering.imfilter)))
Core.eval(BindingAndCatalysis, :(const Kernel = $(ImageFiltering.Kernel)))

_include_visualization("simo_plot.jl")
_include_visualization("graphs.jl")
_include_visualization("rop.jl")
_include_visualization("poly_slices.jl")
_include_visualization("regime_partition.jl")

end
