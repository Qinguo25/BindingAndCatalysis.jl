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