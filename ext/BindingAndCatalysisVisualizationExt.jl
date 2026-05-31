module BindingAndCatalysisVisualizationExt

import BindingAndCatalysis
import GraphMakie
import ImageFiltering
import Latexify
import Makie

const _SRC_DIR = joinpath(dirname(@__DIR__), "src")
const _VIS_DIR = joinpath(_SRC_DIR, "visualization")

function _include_visualization(file::AbstractString)
    return Base.include(BindingAndCatalysis, joinpath(_VIS_DIR, file))
end

# The existing visualization source files are written inside the parent module.
# Import optional plotting names there before including those files.
BindingAndCatalysis.eval(:(using Makie))
BindingAndCatalysis.eval(:(using GraphMakie))
BindingAndCatalysis.eval(:(using GraphMakie.NetworkLayout))
BindingAndCatalysis.eval(:(using Latexify))
BindingAndCatalysis.eval(:(import ImageFiltering: imfilter, Kernel))

_include_visualization("simo_plot.jl")
_include_visualization("graphs.jl")
_include_visualization("rop.jl")
_include_visualization("poly_slices.jl")
_include_visualization("regime_partition.jl")

end
