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
        
        # Compute means manually
        mean_values = sum(projected_data, dims=2) / size(projected_data, 2)
        
        # Ensure the mean of projected data is approximately centered around zero
        @test all(isfinite, mean_values)  # Ensure no NaNs/Infs
        @test all(abs.(mean_values) .< 1e-6)
        
        # Compute variances manually
        var1 = sum((projected_data[1, :] .- mean_values[1]) .^ 2) / (size(projected_data, 2) - 1)
        var2 = sum((projected_data[2, :] .- mean_values[2]) .^ 2) / (size(projected_data, 2) - 1)

        @test isfinite(var1) && isfinite(var2)  # Ensure no NaNs/Infs in variance
        @test var1 >= var2  # First principal component should have the highest variance
    end
    
    @testset "edge cases" begin
        # Single row case - PCA is undefined for a single sample
        single_row = reshape([1.0, 2.0, 3.0], (1, 3))
        @test_throws ArgumentError reduce_to_2d(single_row, 2)
        
        # Single column case (should return the same values since variance is undefined in 1D)
        single_column = reshape([1.0, 2.0, 3.0, 4.0], (4, 1))
        projected_column = reduce_to_2d(single_column, 1)
        @test size(projected_column) == (1, 4)
        @test all(isfinite, projected_column)  # Ensure no NaNs/Infs
    end
end

@testset "show_relations" begin
    # Test data setup
    words = ["king", "queen", "man", "woman", "apple", "banana", "car", "bus"]
    embeddings = [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0;  # First dimension
                  5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0;  # Second dimension
                  9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0]  # Third dimension
    wv = WordEmbedding(words, embeddings)
    
    @testset "even word count enforcement" begin
        @test_throws ArgumentError show_relations("king", "queen", "man"; wv=wv)  # Odd number of words
    end

    @testset "missing words error handling" begin
        @test_throws ArgumentError show_relations("king", "unknown"; wv=wv)  # Word not in embeddings
    end

    @testset "dimension reduction validity" begin
        embeddings_subset = [wv.embeddings[:, wv.word_indices[word]] for word in ["king", "queen", "man", "woman"]]
        reduced = reduce_to_2d(permutedims(hcat(embeddings_subset...)))
        @test size(reduced) == (2, 4)  # Should reduce to 2D space for given words
    end
end
