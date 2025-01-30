using Test
using GroupIWord2Vec

@testset "GroupIWord2Vec" begin
   @testset "Functions" begin
       include("test_functions.jl")
   end
   
   @testset "Model" begin
       include("test_model.jl")
   end

   @testset "Plotting" begin
       include("test_plotting.jl")
   end
end
