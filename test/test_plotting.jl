using Test
using GroupIWord2Vec
using LinearAlgebra    # Needed for eigenvalue/eigenvector validation

@testset "reduce_to_2d Tests" begin
    # Test data: 4 samples, 3 features with a clear pattern
    data = [1.0 2.0 3.0;
            4.0 5.0 6.0;
            7.0 8.0 9.0;
            10.0 11.0 12.0]

    @testset "Basic Functionality & Shape" begin
        reduced = reduce_to_2d(data, 2)
        @test reduced isa Matrix{Float64}
        @test size(reduced) == (2, 4)  # Expecting (number_of_pc × samples)
    end

    @testset "Variance Ordering & Orthogonality" begin
        # High-variance first feature
        data_var = hcat(10 * randn(100, 1), randn(100, 2))  # First column dominates
        reduced_var = reduce_to_2d(data_var, 2)
        @test var(reduced_var[1, :]) ≥ var(reduced_var[2, :])  # Higher variance comes first

        # Orthogonality check
        cov_matrix = cov(data .- mean(data, dims=1))
        _, eigen_vecs = eigen(Symmetric(cov_matrix))
        @test isapprox(eigen_vecs[:, 1] ⋅ eigen_vecs[:, 2], 0; atol=1e-5)
    end

    @testset "Edge Cases & Stability" begin
        @test all(reduce_to_2d(ones(100, 50), 2) .≈ 0)  # Zero-variance case should return zeros
        @test size(reduce_to_2d(randn(100, 1), 1)) == (1, 100)  # Single feature case

        # Tall vs. wide matrices
        @test size(reduce_to_2d(randn(200, 20), 2)) == (2, 200)
        @test size(reduce_to_2d(randn(20, 200), 2)) == (2, 20)

        # Numerical stability: very large/small values
        @test all(isfinite.(reduce_to_2d(randn(100, 50) .* 1e6, 2)))
        @test all(isfinite.(reduce_to_2d(randn(100, 50) .* 1e-6, 2)))
    end
end
