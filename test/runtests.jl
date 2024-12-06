using GroupIWord2Vec
using Test

@testset "GroupIWord2Vec.jl" begin
    @test GroupIWord2Vec.greet_your_package_name() == "Hello GroupIWord2Vec!"
    @test GroupIWord2Vec.greet_your_package_name() != "Hello world!"
end
