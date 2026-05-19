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
        get_one_inner_point(poly::Polyhedron; rand_line=true, rand_ray=true, extend=3, normalize_to_extend=false) -> Vector

Return a point guaranteed to lie inside the polyhedron.

Options:
- `rand_line`: include randomized contribution from linear rays (default: `true`).
- `rand_ray`: randomize scaling of ray directions (default: `true`).
- `extend`: scale factor for ray contributions (default: `3`).
- `normalize_to_extend`: if `true`, the combined ray displacement is normalized
    so its Euclidean norm is approximately `extend` (useful when you want `extend`
    to correspond roughly to distance from `point`). Default: `false`.
"""
function get_one_inner_point(poly::T;rand_line=true,rand_ray=true,extend=3,normalize_to_extend=false) where T<:Polyhedron
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
    # Optionally normalize the total ray displacement so its norm ≈ `extend`.
    if normalize_to_extend && !(norm(ray_avg) ≈ 0)
        ray_avg .= ray_avg ./ norm(ray_avg) .* extend
    end

    return point .+ ray_avg
end
