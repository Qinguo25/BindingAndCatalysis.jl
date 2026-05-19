@testset "Constraint Chain Length" begin
    C = [
        1 -1 0
        0 1 -1
        4 0 -2
    ]

    result = calc_chainlength(C)
    @test result.chainlength == 2
    @test result.sources == Set([1, 2])
    @test result.sinks == Set([2, 3])
    @test result.source_only == Set([1])
    @test result.sink_only == Set([3])
    @test result.both == Set([2])

    cyclic = calc_chainlength([1 -1; -1 1])
    @test cyclic.chainlength == Inf
    @test isempty(cyclic.source_only)
    @test isempty(cyclic.sink_only)
    @test cyclic.both == Set([1, 2])
end
