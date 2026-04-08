module NativePolyhedra

using LinearAlgebra
using SparseArrays
using JuMP
import Clarabel
import MathOptInterface as MOI

import Base: +, -, *, /, ==, hash, show, zero, iszero, convert, promote_rule, Float64, BigFloat, isless, ^, isempty, in, intersect, issubset, float, abs, abs2, real, conj, <, <=, >, >=

export ExactLogExpr, exact_log10, exact_log10_ratio
export Polyhedron, HRep, MixedMatHRep, hrep, polyhedron
export HalfSpace, HyperPlane, intersect, eliminate, detecthlinearity!, removehredundancy!
export dim, fulldim, hashyperplanes, hyperplanes, allhalfspaces, issubset
export feasible_point, interior_point

include("exact_types.jl")
include("polyhedra_core.jl")

end
