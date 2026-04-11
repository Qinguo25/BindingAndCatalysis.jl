"""
    arr_to_vector(arr)

Convert a multidimensional array into nested vectors.
"""
function arr_to_vector(arr)
    d = ndims(arr)
    if d == 0
        return arr[]
    elseif d == 1
        return [x for x in arr]
    else
        return [arr_to_vector(s) for s in eachslice(arr, dims=1)]
    end
end

"""
    pythonprint(arr) -> nothing

Pretty-print an array in JSON format for inspection.
"""
function pythonprint(arr)
    txt = JSON3.write(arr_to_vector(arr), pretty=true, indent=4, escape_unicode=false)
    println(txt)
    return nothing
end

"""
    _ode_solution_wrapper(solution::ODESolution) -> (Vector{Float64}, Vector{Vector{Float64}})

Convert a DifferentialEquations solution into time and state arrays.
"""
function _ode_solution_wrapper(solution::ODESolution)::Tuple{Vector{Float64}, Vector{Vector{Float64}}}
    return solution.t, solution.u
end
