using BindingAndCatalysis
using Polyhedra
using Random

function subset_poly(P, Q)
    all(h -> issubset(P, h), allhalfspaces(hrep(Q)))
end

function print_poly(io, name, poly, syms)
    C, C0, n = get_C_C0_nullity(poly)
    println(io, "=== ", name, " ===")
    println(io, "nullity = ", n)
    println(io, "C =")
    show(io, "text/plain", Matrix(C))
    println(io)
    println(io, "C0 = ", collect(C0))
    println(io, "conditions:")
    for cond in show_condition_poly(poly; syms=syms)
        println(io, cond)
    end
    println(io)
end

function reduced(poly)
    out = eliminate(poly, 1)
    detecthlinearity!(out)
    removehredundancy!(out)
    return out
end

function find_witness(better_poly, siso_poly; seed = 1, ntries = 200_000)
    Random.seed!(seed)
    Cb, C0b, _ = get_C_C0_nullity(better_poly)
    Cs, C0s, _ = get_C_C0_nullity(siso_poly)
    for _ in 1:ntries
        x = rand(4) .* 4 .- 2
        better_ok = all(Cb * x .+ C0b .> 1e-8)
        siso_ok = all(Cs * x .+ C0s .> 1e-8)
        if better_ok && !siso_ok
            return x, Cb * x .+ C0b, Cs * x .+ C0s
        end
    end
    return nothing
end

function main()
    N = [1 1 -1 0 0; 1 0 1 -1 0; 0 1 0 1 -1]
    model = Bnc(N = N)
    path = [1, 4, 3]
    siso = SISOPaths(model, 1)
    syms = qK_sym(siso)
    siso_poly = get_polyhedron(siso, path)

    helper = BindingAndCatalysis.SISOHelper(model, 1)
    BindingAndCatalysis._find_all_path_conditions!(helper)
    helper_poly = nothing
    for source in helper.sources, sink in helper.sinks
        ps = helper.paths[source, sink]
        ps === nothing && continue
        for p in ps
            if p.path == path
                helper_poly = p.condition
                break
            end
        end
        helper_poly === nothing || break
    end
    helper_poly === nothing && error("Path $(path) not found in SISOHelper results.")

    prism1 = BindingAndCatalysis._get_polyhedron_prism(model, 1, 1)
    prism4 = BindingAndCatalysis._get_polyhedron_prism(model, 4, 1)
    prism3 = BindingAndCatalysis._get_polyhedron_prism(model, 3, 1)
    interface14 = BindingAndCatalysis._get_interface_prism(model, 1, 4, 1)
    interface43 = BindingAndCatalysis._get_interface_prism(model, 4, 3, 1)

    vertex_path = intersect(prism1, prism4, prism3)
    detecthlinearity!(vertex_path)
    removehredundancy!(vertex_path)
    interface_path = intersect(interface14, interface43)
    detecthlinearity!(interface_path)
    removehredundancy!(interface_path)
    same = same_polyhedron(siso_poly, helper_poly)
    witness = same ? nothing : find_witness(helper_poly, siso_poly)

    println("same_polyhedron(siso, helper) = ", same)
    println("siso_subset_helper = ", subset_poly(siso_poly, helper_poly))
    println("helper_subset_siso = ", subset_poly(helper_poly, siso_poly))
    println("helper_equals_vertex_prisms = ", same_polyhedron(helper_poly, vertex_path))
    println("siso_equals_interface_prisms = ", same_polyhedron(siso_poly, interface_path))
    if witness === nothing
        println("witness = not found")
    else
        x, helper_margin, siso_margin = witness
        println("witness_log10_[q2,K1,K2,K3] = ", collect(x))
        println("witness_helper_margins = ", collect(helper_margin))
        println("witness_siso_margins = ", collect(siso_margin))
    end
    println()

    print_poly(stdout, "SISO path [1,4,3]", siso_poly, syms)
    print_poly(stdout, "helper path [1,4,3]", helper_poly, syms)
    print_poly(stdout, "vertex prism 1", prism1, syms)
    print_poly(stdout, "vertex prism 4", prism4, syms)
    print_poly(stdout, "vertex prism 3", prism3, syms)
    print_poly(stdout, "interface prism 1-4", interface14, syms)
    print_poly(stdout, "interface prism 4-3", interface43, syms)
end

main()
