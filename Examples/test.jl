using Pkg
dev_folder = "./Examples/"
pkg_folder = "./"
Pkg.activate(dev_folder)
Pkg.develop(path=pkg_folder)


using BindingAndCatalysis
using Revise
using GLMakie

N = [1 1 -1]
model = Bnc(N=N)