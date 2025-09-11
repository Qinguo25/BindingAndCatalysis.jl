using DifferentialEquations
using Plots
# case 1
f(u, p, t) = 0.98u 
u0 = 1.0
tspan = (0.0, 10.0)
prob = ODEProblem(f, u0, tspan)
sol = solve(prob)


# case 1 (array form)
f!(du,u,p,t) = du .= 0.98 .* u
u0 = [1.0, 0.3]
tspan = (0.0, 10.0)
prob = ODEProblem(f!, u0, tspan)
sol = solve(prob ; alg=nothing;)


# case 2
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
    # return nothing  # du is modified in place, so no need to return anything
end

u0 = [1.0,0,0,0]
p = [10.0, 28.0, 8/3]
tspan = (0.0, 100.0)
prob = ODEProblem(lorenz!, u0, tspan, p)
sol = solve(prob)


#----------
sol.t[10], sol[10] # Output the time and solution at the 10th step
sol[2,10] # Output the second variable at the 10th step
convert(Array, sol) # Convert the solution to an array
plot(sol, idxs=(1,2,3))
t = 1.0:0.01:100.0
u = sol(t)

fig = Figure()
lines!(Axis3(fig[1,1]), u[1,:], u[2,:], u[3,:], label="Lorenz attractor")

line
# Plot the first three variables of the solution
plot(sol, idxs=(1,2,3), denseplot=false)
#----------



#case 3
function lotka_volterra!(du,u,p,t)
    du[1] = p[1] * u[1] - p[2] * u[1] * u[2]
    du[2] = -p[3] * u[2] + p[4] * u[1] * u[2]
end

using ParameterizedFunctions # (big package)
lv! = @ode_def LotkaVolterra begin
    dx = a*x - b*x*y
    dy = -c*y + d*x*y
end a b c d 
#(but with a waring tell you to define "@independent_variables t")
# many benifits including: calculationg Jacobian and can be easily export by Latexify.jl, and inherently speed up by Jacobian

u0 = [1.0, 1.0]
p = [1.5, 1., 3., 1.]
tspan = (0.0, 10.0)
prob = ODEProblem(lv!, u0, tspan, p)
sol = solve(prob)

plot(sol)


#case 4 Matrix

A = rand(4,4)

u0 = rand(4,2)
tspan = (0.0, 1.)
f!(du,u,p,t) = du .= A * u
prob = ODEProblem(f!, u0, tspan)
sol = solve(prob)

# change the type of u0 to BigFloat
big_u0 = big.(u0)
prob = ODEProblem(f!, big_u0, tspan)
sol = solve(prob)

## using Static array will speedup code for small matrix
using StaticArrays
A = @SMatrix rand(4,4)
u0 = @SMatrix rand(4,2)
tspan = (0.0, 1.)
f(u,p,t) = A * u
prob = ODEProblem(f, u0, tspan)
sol = solve(prob)
#"If using a immutable initial condition type, please use the out-of-place form.
#"I.e. define the function `du=f(u,p,t)` instead of attempting to "mutate" the immutable `du`.""


#case 5

van! = @ode_def VanDerPol begin
    dy = μ*((1-x^2)*y - x)
    dx = y
end μ

prob = ODEProblem(van!, [0.0, 2.0], (0.0, 6.3), 1e6)


using BenchmarkTools

@btime sol = solve(prob)
@btime sol = solve(prob,alg_hints = [:stiff])

plot(sol,denseplot=false, ylims=(-10,10))




struct MyStruct
    x::Int
    y::Vector{Int}
end

a = [1, 2, 3]
s1 = MyStruct(42, a)
s2 = MyStruct(42, a)

println(s1.y === s2.y)  # true，说明 s1.y 和 s2.y 引用同一个数组
a[1] = 100
println(s1.y)  # [100, 2, 3]，s1.y 受影响
println(s2.y)  # [100, 2, 3]，s2.y 也受影响

# using Setfield

# struct BigStruct
#     a::Int
#     b::Float64
#     c::String
#     d::Vector{Int}
# end

# s1 = BigStruct(1, 2.0, "hello", [10, 20])

# # Update fields 'a' and 'c' at the same time
# s1 = setproperties(s1, (a = 99, c = "world"))

# println("Original: ", s1)
# println("Updated:  ", s2)

bc! = @ode_def bounceball begin
    dy = v
    dv = -g
end g






 # --------------------------------------Call backs--------------------------------------
# ContinuousCallback
condition = (u,t,integrator)->u[1]
affect! = (integrator)->begin
    integrator.u[2] = -integrator.u[2] * integrator.p[2]
    integrator.p[2] = 1-sqrt(1-integrator.p[2]) # bounce with 90% energy loss
end
bounce_cb = ContinuousCallback(condition, affect!)

#DiscreteCallback
condition_kick = (u,t,integrator) -> t==2
affect_kick! = (integrator) -> begin
    integrator.u[2] += 50 # kick the ball up
end
kick_cb = DiscreteCallback(condition_kick, affect_kick!)

#terminateevents
condition_terminate = (u,t,integrator) -> u[1]+1e-6
terminate_cb = ContinuousCallback(condition_terminate, terminate!)
cb = CallbackSet(bounce_cb, kick_cb,terminate_cb)

let 
    u0 = [50., 0.]
    p = [9.81, 0.9] # g, energy loss factor
    tspan = (0.0, 40.0)
    prob = ODEProblem(bc!, u0, tspan, p; callback = cb, tstops = [2.0])
    sol = solve(prob)
    plot(sol)
end


# energy preserve
let
    function g(resid,u,p,t)
        resid[1] = u[1]^2 + u[2]^2 - 1.0
        resid[2]
    end
    cb = ManifoldProjection(g)
    sol = solve(prob, callback = cb)
end
#Positive Call back
PositiveCallback()

#trace of matrix a

# Saving Callback
prob = ODEProblem((du,u,p,t) -> du .= u, rand(1000,1000), (0.,1.))
saved_values = SavedValues(Float64, Tuple{Float64,Float64})
cb = SavingCallback((u,t,integrator)->(tr(u),norm(u)),saved_values, saveat=0.0:0.1:1.0)
sol = solve(prob, callback = cb, save_everystep=false, save_start=false, save_end=false)
saved_values.t
saved_values.saveval

#-------------------------------end of call backs--------------------------------------

#---------test--------
using LinearAlgebra
let
    A = rand(3, 3)
    B = rand(3)
    # 2. Pre-allocate a vector to store the solution x
    x = zeros(3)
    println("x before ldiv!: ", x)
    # 3. Perform the in-place solve into x
    # This solves A*x=B and stores the result in our pre-allocated x
    # A and B are not modified.
    ldiv!(x, A, B)
    println("x after ldiv!:  ", x)
    # 4. Verify the result
    @assert x ≈ A \ B
end
#-------end test--------

#Using Symbolics to solve Equations

using Symbolics

@variables A B C CAB CAC CABC tA tB tC;
@variables k1 k2 k3 k4 k1_ k2_ k3_ k4_; 

cons1 = A + CAB + CAC + CABC - tA
cons2 = B + CAB + CABC - tB
cons3 = C + CAC + CABC - tC
ss1 = k1 * A * B - k1_ * CAB + k2_ * CABC - k2 * CAB * C
ss2 = k3 * A * C - k3_ * CAC + k4_ * CABC - k4 * CAC * B
ss3 = k2 * CAB * C + k4 * CAC * B - (k2_ + k4_) * CABC
eqs = [cons1 ~ 0, cons2 ~ 0, cons3 ~ 0, ss1 ~ 0, ss2 ~ 0, ss3 ~ 0]
vars = [A, B, C, CAB, CAC, CABC]
sol = symbolic_solve(eqs, vars)


mutable struct bnc2
    # Parameters of the binding networks
    N::Matrix{Int} # binding reaction matrix
    L::Matrix{Int} # conservation law matrix

    r::Int # number of reactions
    n::Int # number of variables
    d::Int # number of conserved quantities

    x_sym::Union{Vector{Symbol},Nothing} # species symbols, each column is a species
    q_sym::Union{Vector{Symbol},Nothing}
    K_sym::Union{Vector{Symbol},Nothing}


    # Parameters for the catalysis networks
    S::Union{Matrix{Int},Nothing} # catalysis change in qK space, each column is a reaction
    aT::Union{Matrix{Int},Nothing} # catalysis index and coefficients, rate will be vⱼ=kⱼ∏xᵢ^aT_{j,i}
    
    k::Union{Vector{Float64},Nothing} # rate constants for catalysis reactions
    
    r_cat:: Union{Int,Nothing} # number of catalysis reactions/species



    # Parameters act as the starting points used for qk mapping
    _anchor_log_x::Vector{Float64}
    _anchor_log_qK::Vector{Float64}
    #Parameters for mimic calculation process
    _is_change_of_K_involved::Bool  # whether the K is involved in the calculation process

    # Inner constructor 
    function bnc2(N, L, x_sym, q_sym, K_sym, S, aT, k)
        # get desired values
        r, n = size(N)
        d, n_L = size(L)

        # Validate dimensions for binding network, check if its legal.
        @assert n == d+r "d+r is not equal to n"
        @assert n_L == n "L must have the same number of columns as N"
        
        isnothing(x_sym) || @assert length(x_sym) == n "x_sym length must equal number of species (n)"
        isnothing(q_sym) || @assert length(q_sym) == d "q_sym length must equal number of conserved quantities (d)"
        isnothing(K_sym) || @assert length(K_sym) == r "K_sym length must equal number of reactions (r)"


        # Validate dimensions for catalysis network
        #check if catalysis networks paramets legal
        if ~isnothing(S)
            n_S, r_cat = size(S)
            @assert n_S == n "S must have the same number of columns as N"
        else
            r_cat = nothing
        end

        if ~isnothing(aT)
            r_aT, n_aT = size(aT)
            if ~isnothing(r_cat)
                @assert r_aT == r_cat "aT must have the same number of rows as r_cat"
            else
                r_cat = r_aT
            end
            @assert n_aT == n "aT must have the same number of rows, as columns of N"
        end

        if ~isnothing(k)
            @assert isnothing(r_cat) || length(k) == r_cat "k must have the same length as r_cat"
        end

        #helper parameters 
        _anchor_log_x = zeros(n)
        _anchor_log_qK = vcat(vec(log.(sum(L; dims=2))), zeros(r))
        _is_change_of_K_involved = S === nothing || all(@view(S[r+1:end, :]) .== 0)
        # Create the new object with all fields specified
        new(N, L, r, n, d,  x_sym, q_sym, K_sym, S, aT, k , r_cat, _anchor_log_x, _anchor_log_qK, _is_change_of_K_involved)
    end
end


function bnc2(; 
    N::Matrix{Int}, 
    L::Union{Matrix{Int},Nothing}=nothing, 
    x_sym::Union{Vector{Symbol},Nothing}=nothing, 
    q_sym::Union{Vector{Symbol},Nothing}=nothing, 
    K_sym::Union{Vector{Symbol},Nothing}=nothing,
    S::Union{Matrix{Int},Nothing}=nothing, 
    aT::Union{Matrix{Int},Nothing}=nothing, 
    k::Union{Vector{Float64},Nothing}=nothing
    )::bnc2

    if isnothing(L)
        L = L_from_N(N)
        end
     # Call the inner constructor
     # Number of variables in the binding network
    
     #fufill S matrix
    if ~isnothing(S)
        n = size(N, 2)
        (nrow_S,ncol_S) = size(S)
        if nrow_S < n
            # If S is not provided or has fewer columns than n, fill it with zeros, make sure S has n rows.
            S = vcat(S, zeros(Int64, n - nrow_S, ncol_S))
        end
    end
    
    # @show N,L,x_sym,q_sym,K_sym,S,aT,k
    bnc2(N, L, x_sym, q_sym, K_sym, S, aT, k )
end

function L_from_N(N::Matrix{Int})::Matrix{Int}
    r, n = size(N)
    d = n - r
    N_1 = @view N[:,1:d]
    N_2 = @view N[:,d+1:n]
    hcat(Matrix(I, d, d), -(N_2 \ N_1)')
end

N2 = [ 1  1  0 -1  0
     1  0  1  0 -1]

bnc_test2_1 = bnc2(N=N2)

function testtime1(t::bnc)
   t.N
end
function testtime1(t::bnc2)
   t.N
end


using Catalyst

rn = @reaction_network begin
    (k1,k2), Cumate + cymR <--> C1
    (k3,k4), cymR + DNA <--> cymR_DNA
    (k3,k4), cymR + DNA_RNP <--> cymR_DNA_RNP
    (k5,k6), RNP + DNA <--> DNA_RNP
    (k5,k6), RNP + cymR_DNA <--> cymR_DNA_RNP
    k7, DNA_RNP --> gRNA
    k8, DNA_RNP --> dCpf1
    k9, DNA_RNP --> GFP
    (k9,k10), gRNA + dCpf1 <--> gRNA_dCpf1
    (k11,k12), gRNA_dCpf1 + DNA2 <--> gRNA_dCpf1_DNA2
    (k11,k12), gRNA_dCpf1 + DNA2_RNP <--> gRNA_dCpf1_DNA2_RNP
    (k13,k14), RNP + DNA2 <--> DNA2_RNP
    (k13,k14), RNP + gRNA_dCpf1_DNA2 <--> gRNA_dCpf1_DNA2_RNP
    k15, DNA2_RNP --> RNP
end





function construct_M_from_c(L::Matrix{Int}, c)
    d, n = size(L)
    M = zeros(Int, d, n)
    for i in 1:d
        M[i, c[i]] = L[i, c[i]]
    end
    return M
end

# using LinearAlgebra


using Graphs # Assuming the use of the Graphs.jl library

# The core is_cyclic function for directed graphs from your example
function has_cycle_from_nodes(g::AG, start_nodes) where {T,AG<:AbstractGraph{T}}
    # 0: unvisited, 1: visiting (in recursion stack), 2: visited
    vcolor = zeros(UInt8, nv(g))
    vertex_stack = Vector{T}()

    # --- MODIFICATION IS HERE ---
    # Instead of iterating over all vertices, iterate only over the provided start_nodes.
    for v in start_nodes
        # If we have already visited this node from a previous starting node's traversal, skip it.
        vcolor[v] != 0 && continue

        push!(vertex_stack, v)
        while !isempty(vertex_stack)
            u = vertex_stack[end]
            if vcolor[u] == 0
                vcolor[u] = 1 # Mark as visiting
                for n in outneighbors(g, u)
                    # Cycle detected: found a back edge to a node in the current path
                    if vcolor[n] == 1
                        return true
                    elseif vcolor[n] == 0
                        push!(vertex_stack, n)
                    end
                end
            else # This block is reached when we are backtracking
                pop!(vertex_stack)
                if vcolor[u] == 1
                    vcolor[u] = 2 # Mark as fully explored
                end
            end
        end
    end
    
    # If we finish all traversals from start_nodes without finding a cycle
    return false
end

using Graphs
# using IterTools


function find_possible_c(L::Matrix{Int})
    # Get dimensions of the input matrix L
    d, n = size(L)
    # Step 1: For each row i, find indices j where L[i, j] > 0
    J = Vector{Vector{Int}}()
    for i in 1:d
        J_i = [j for j in 1:n if L[i, j] > 0]
        push!(J, J_i)
    end
    
    # Step 2: Generate all possible selections S using Cartesian product
    all_c = Iterators.product(J...)
    
    # Step 3: Initialize list to store all valid M matrices
    # feasible_M = Vector{Matrix{Int}}()
    feasible_c = Vector{Tuple}()

    for c in all_c
        # Step 4: Build the directed graph for this selection
        G = SimpleDiGraph(n)
        for i in 1:d
            c_i = c[i]
            # Add edges from c_i to all m ≠ c_i where L[i, m] > 0
            for m in J[i]
                if m != c_i
                    add_edge!(G, c_i, m)
                end
            end
        end
        
        # Step 5: Check if the graph is acyclic
        if !is_cyclic(G)
            push!(feasible_c, c)
        end
    end
    return feasible_c
    # return feasible_M
end

using BenchmarkTools



G = SimpleDiGraph(1000000)
ict = IncrementalCycleTracker(G, dir= :in)
for i in 1:999999
    add_edge!(G,i+1,i)
end


@btime let
    for i in 3:10000
        add_edge_checked!(ict, i,1)
    end
end #357us #4.2771ms #1.813ms

@btime let
        add_edge_checked!(ict, 3:10000, 1)
end

#144us # 1.973ms #2ms
@btime let
    for i in 3:10000
        add_edge!(G,i,1)
    end
    has_cycle_from_nodes(G,1)
end 



G = SimpleDiGraph(5)
ict = IncrementalCycleTracker(G, dir= :out)
for i in 1:2
    add_edge!(G,i,i+1)
end
add_edge!(G,1,5)
G
add_edge_checked!(ict, 5, [1,4])


# using Graphs
# function find_valid_regime(L::Matrix{Int})
#     (d,n) = size(L)
#     idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L) ] #!!! extremely key, avoid repeated add, or bug when removing it.
#     graph = SimpleDiGraph(n)
#     ict = IncrementalCycleTracker(graph, dir = :out)
#     choices = Vector{Int}(undef, d)
#     results = Vector{Vector{Int}}()
#     function backtrack!(i)
#         if i == d+1 
#             push!(results, copy(choices))  # 使用副本避免后续修改影响结果
#             return nothing
#         end

#         for v in idx[i]
#             target_nodes = [w for w in idx[i] if w != v && ~(w in outneighbors(graph,v))] # target_nodes
#             if add_edge_checked!(ict, v, target_nodes) # add successfully
#                 choices[i] = v
#                 backtrack!(i + 1)
#             end

#             for node in target_nodes
#                 rem_edge!(graph, v, node)
#             end
#         end
#     end
#     backtrack!(1)
#     return results
# end




G = SimpleDiGraph(4)
ict = IncrementalCycleTracker(G, dir= :out)
add_edge_checked!(ict, 1,[3,4])
add_edge_checked!(ict, 2,[3,4])
rem_edge!(G,2,3)
rem_edge!(G,2,4)



# Example usage
L = [1 0 1 1; 0 1 1 1]

find_valid_regime(L)


@btimefind_possible_c(L)
# Find all possible M matrices
find_possible_c(L)



#----------play ground
function is_cyclic(g::Vector{Vector{Int}}, node::Int, len::Int) 
    # 0 if not visited, 1 if in the current dfs path, 2 if fully explored
    vcolor = zeros(UInt8, len)
    vertex_stack = Vector{Int8}()
    # vcolor[node] != 0 && continue
    push!(vertex_stack, node)
    while !isempty(vertex_stack)
        u = vertex_stack[end]
        if vcolor[u] == 0
            vcolor[u] = 1
            for n in g[u]
                # we hit a loop when reaching back a vertex of the main path
                if vcolor[n] == 1
                    return true
                elseif vcolor[n] == 0
                    # we store neighbors, but these are not yet on the path
                    push!(vertex_stack, n)
                end
            end
        else
            pop!(vertex_stack)
            if vcolor[u] == 1
                vcolor[u] = 2
            end
        end
    end
    return false
end
function find_valid_regime(L::Matrix{Int})
    (d,n) = size(L)
    idx = [[idx for (idx, value) in enumerate(row) if value != 0] for row in eachrow(L) ]
    graph = [Vector{Int}() for _ in 1:n]
    choices = Vector{Int}(undef, d)
    results = Vector{Vector{Int}}()
    function backtrack!(i)
        # All rows are fine
        if i == d+1 
            # @show choices
            push!(results, copy(choices)) 
            return nothing
        end

        for v in idx[i]
            # add edges for current row. and record.

            target_nodes = [w for w in idx[i] if w != v ] # target_nodes
            
            for node in target_nodes
                push!(graph[node], v) # add edge node -> v
            end

            if  ~is_cyclic(graph, v, n)
                choices[i] = v
                backtrack!(i + 1)
            end

            for node in target_nodes
                pop!(graph[node])
            end
        end
    end
    backtrack!(1)
    return results
end

function idx_to_vertex(L::Matrix{Int}, idx::Vector{Int})
    M = zeros(Int,size(L))
    for (i,j) in enumerate(idx)
        M[i,j] = 1
    end
    return M
end

using Graphs
function graph_generator(L)
    d, n = size(L)
    M1 = [zeros(Int, d, d)  L ; L' zeros(Int, n, n)]
    G = SimpleGraph(M1)
    return G
end

#----------

# Print the results
println("Number of possible M matrices: ", length(matrices_M))
for (idx, M) in enumerate(matrices_M)
    println("\nM_$idx:")
    println(M)
end



using GLMakie, GraphMakie
using GraphMakie.NetworkLayout
# - ------

# L = [1 0 0 1 1 0 ; 0 1 0 1 0 1 ; 0 0 1 0 1 1]
# L = [1 0 1 1; 0 1 1 1]
# L = L_generator(10,20,min_binder=2,max_binder=2) 
L = L_generator(10,20)
# Example matrix
# valids = find_valid_regime(L)
let
d,n = size(L)
# M = idx_to_vertex(L, valids[16])
# G = graph_generator(M)
G = graph_generator(L)

p = graphplot(G; layout = Stress(;dim=3),
    node_size = 20 , node_color = vcat([:blue for i in 1:d],[:red for i in 1:n]), edge_color = :black,
    nlabels = vcat(["U"*string(i) for i in 1:d],["V"*string(i) for i in 1:n]),
    # edge_label = [string(i) for i in 1:ne(G)],
    # camera = (zoom=0.5, elevation=0.5, azimuth=0.5),
    # axis = (aspect=DataAspect(), title="Graph Plot")
)
display(p)
# M
end