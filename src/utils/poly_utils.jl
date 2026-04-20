function same_polyhedron(P, Q)
    fulldim(P) == fulldim(Q) || return false

    HP = hrep(P)
    HQ = hrep(Q)

    return all(h -> issubset(P, h), allhalfspaces(HQ)) &&
           all(h -> issubset(P, h), hyperplanes(HQ)) &&
           all(h -> issubset(Q, h), allhalfspaces(HP)) &&
           all(h -> issubset(Q, h), hyperplanes(HP))
end
