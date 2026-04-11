function _normalized_constraint_signature(rep, i::Int)
    sig = NativePolyhedra._signed_signature(rep.A, rep.b, i)
    return i in rep.linset ? NativePolyhedra._unsigned_signature(sig) : sig
end

function _hrep_signature(rep)
    sigs = [_normalized_constraint_signature(rep, i) for i in 1:size(rep.A, 1)]
    sort!(sigs, by=string)
    return Tuple(sigs)
end

function same_polyhedron(P, Q)
    HP = hrep(P)
    HQ = hrep(Q)

    size(HP.A, 2) == size(HQ.A, 2) || return false
    _hrep_signature(HP) == _hrep_signature(HQ) && return true

    all(h -> issubset(P, h), allhalfspaces(HQ)) &&
    all(h -> issubset(P, h), hyperplanes(HQ)) &&
    all(h -> issubset(Q, h), allhalfspaces(HP)) &&
    all(h -> issubset(Q, h), hyperplanes(HP))
end
