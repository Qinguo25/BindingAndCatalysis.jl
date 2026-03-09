begin
    using Pkg
    dev_folder = "./Examples/" # folder of the development environment
    # pkg_folder = "./" # folder of the package
    Pkg.activate(dev_folder)
    # Pkg.develop(path=pkg_folder)
end
Threads.nthreads() 

using Revise
using BindingAndCatalysis # import the package
# using CairoMakie

N = [2 1 -1]
model = Bnc(N=N)

show_condition(model,1)
show_condition(model,2)
show_condition(model,3)
show_condition(model,4)


#------------------
# test
#-----------------
using GLMakie
using GeometryBasics
using LinearAlgebra

# -----------------------------
# Rays (same as before)
# -----------------------------
V = [
    Vec3f( 1,  0,  0),  # v1
    Vec3f( 0,  1,  0),  # v2
    Vec3f( 0,  0,  1),  # v3
    Vec3f( 0, -1, -1),  # v4
    Vec3f(-1,  0, -1),  # v5
    Vec3f(-2, -1,  0)   # v6
]

# Maximal cones <vi, vj, vk> (indices into V)
max_cones = [
    (1, 2, 3),
    (1, 2, 4),
    (2, 4, 5),
    (2, 3, 5),
    (3, 5, 6),
    (1, 3, 6),
    (1, 4, 6),
    (4, 5, 6)
]

# -----------------------------
# Normalize rays to the unit sphere S^2
# (use Point3f for mesh vertices)
# -----------------------------
P = Point3f[]
for v in V
    vn = v / norm(v)
    push!(P, Point3f(vn))
end

faces = TriangleFace{Int}[]
for (i, j, k) in max_cones
    push!(faces, TriangleFace(i, j, k))
end
fan_mesh = GeometryBasics.Mesh(P, faces)

# -----------------------------
# Optional: a translucent unit sphere for reference
# -----------------------------
function sphere_mesh(radius::Float32 = 1f0; nθ::Int=48, nφ::Int=24)
    pts = Point3f[]
    for j in 0:nφ
        φ = Float32(pi) * (j / nφ)              # 0..π
        for i in 0:nθ
            θ = 2f0 * Float32(pi) * (i / nθ)    # 0..2π
            x = radius * cos(θ) * sin(φ)
            y = radius * sin(θ) * sin(φ)
            z = radius * cos(φ)
            push!(pts, Point3f(x, y, z))
        end
    end

    idx(i, j) = j*(nθ+1) + i + 1  # 1-based indexing into pts
    fs = TriangleFace{Int}[]
    for j in 0:nφ-1
        for i in 0:nθ-1
            a = idx(i,   j)
            b = idx(i+1, j)
            c = idx(i,   j+1)
            d = idx(i+1, j+1)
            push!(fs, TriangleFace(a, b, c))
            push!(fs, TriangleFace(b, d, c))
        end
    end
    return GeometryBasics.Mesh(pts, fs)
end

S = sphere_mesh(1f0)

# -----------------------------
# Plot
# -----------------------------
fig = Figure(size = (1000, 800))
ax = Axis3(fig[1, 1], aspect = :data, xlabel = "x", ylabel = "y", zlabel = "z")

# Sphere (very light, to give context)
mesh!(ax, S; transparency = true, alpha = 0.10)

# Fan spherical triangles
mesh!(ax, fan_mesh; transparency = true, alpha = 0.45)
wireframe!(ax, fan_mesh; linewidth = 2.0)

# Vertices on the sphere + labels
scatter!(ax, P; markersize = 18)
for (idx, p) in enumerate(P)
    text!(ax, "v$idx"; position = Point3f(p) .+ Vec3f(0.04, 0.04, 0.04), fontsize = 18)
end

# Great-circle-ish rays from origin to each point (optional)
O = Point3f(0, 0, 0)
for p in P
    lines!(ax, [O, p]; linewidth = 3.0)
end

ax.azimuth[] = 0.9
ax.elevation[] = 0.35
fig



