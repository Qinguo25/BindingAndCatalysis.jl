using JuliaFormatter

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))

format(joinpath(ROOT, "src"))
format(joinpath(ROOT, "test"))
