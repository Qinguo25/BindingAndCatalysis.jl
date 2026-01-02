begin
    using Pkg
    dev_folder = "./Examples/" # folder of the development environment
    pkg_folder = "./" # folder of the package
    Pkg.activate(dev_folder)
    Pkg.develop(path=pkg_folder)
end
Threads.nthreads() 

using BindingAndCatalysis # import the package
using CairoMakie

N = [1 1 -1]
model = Bnc(N=N)