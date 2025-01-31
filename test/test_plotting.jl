using Test
using GroupIWord2Vec

@testset "reduce_to_2d" begin
    # Generate synthetic data with a clear pattern
    data = [1.0 2.0 3.0;
            4.0 5.0 6.0;
            7.0 8.0 9.0;
            10.0 11.0 12.0]
    
    @testset "basic functionality" begin
        projected_data = reduce_to_2d(data, 2)
        @test projected_data isa Matrix{Float64}  # Ensure correct return type
        @test size(projected_data) == (2, 4)  # Ensure correct shape after projection
    end
    
    @testset "dimensionality reduction" begin
        projected_data = reduce_to_2d(data, 1)
        @test size(projected_data) == (1, 4)  # Ensure correct shape for 1D reduction
    end
    
    @testset "numerical properties" begin
        projected_data = reduce_to_2d(data, 2)
        
        # Ensure the mean of projected data is approximately centered around zero
        @test all(abs.(mean(projected_data, dims=2)) .< 1e-6)
        
        # Ensure variance along principal components is decreasing
        var1 = var(projected_data[1, :])
        var2 = var(projected_data[2, :])
        @test var1 >= var2  # First principal component should have the highest variance
    end
    
    @testset "edge cases" begin
        # Single row case (should return the same row after centering)
        single_row = reshape([1.0, 2.0, 3.0], (1, 3))
        projected_single = reduce_to_2d(single_row, 2)
        @test size(projected_single) == (2, 1)
        
        # Single column case (should reduce to itself)
        single_column = reshape([1.0, 2.0, 3.0, 4.0], (4, 1))
        projected_column = reduce_to_2d(single_column, 1)
        @test size(projected_column) == (1, 4)
    end
end
