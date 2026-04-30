export get_one_inner_point


function same_polyhedron(P, Q)
    fulldim(P) == fulldim(Q) || return false

    HP = hrep(P)
    HQ = hrep(Q)

    return all(h -> issubset(P, h), allhalfspaces(HQ)) &&
           all(h -> issubset(P, h), hyperplanes(HQ)) &&
           all(h -> issubset(Q, h), allhalfspaces(HP)) &&
           all(h -> issubset(Q, h), hyperplanes(HP))
end




"""
    get_one_inner_point(poly::Polyhedron; rand_line=true, rand_ray=true, extend=3) -> Vector

Return a point guaranteed to lie inside the polyhedron.
"""
function get_one_inner_point(poly::T;rand_line=true,rand_ray=true,extend=3) where T<:Polyhedron
    vrep_poly = MixedMatVRep(vrep(poly))
    point = if size(vrep_poly.V, 1) == 0
        zeros(Float64, fulldim(poly))
    else
        [mean(col) for col in eachcol(vrep_poly.V)]
    end
    ray_avg = zeros(eltype(point), length(point))
    for (i, ray) in enumerate(eachrow(vrep_poly.R))
        if i ∉ vrep_poly.Rlinset
            norm_ray = norm(ray)
            sigma = rand_ray ? (rand() + 0.5) * extend : extend
            ray_avg .+= ray ./ norm_ray .* sigma
        elseif rand_line
            norm_ray = norm(ray)
            sigma = (rand() - 0.5) * extend
            ray_avg .+= ray ./ norm_ray .* sigma
        end
    end
    return point .+ ray_avg
end
