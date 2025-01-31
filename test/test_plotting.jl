using Test
using GroupIWord2Vec

# INFO: We hade problems using the Statistics or LinearAlgebra package here which is why we computed e.g. the mean/variance manually instead of using a predefined funtion

@testset "reduce_to_2d" begin
    data = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0; 10.0 11.0 12.0]
    
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
    words = ["king", "queen", "man", "woman", "apple", "banana", "car", "bus"]
    embeddings = [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0; 5.0 6.0 7.0 8.0 9.0 10.0 11.0 12.0; 9.0 10.0 11.0 12.0 13.0 14.0 15.0 16.0]
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

    @testset "arrow computation" begin
        selected_words = ["king", "queen", "man", "woman"]
        projection_test = reduce_to_2d(permutedims(hcat([wv.embeddings[:, wv.word_indices[word]] for word in selected_words]...)))

        word_count = length(selected_words)
        arrows = [projection_test[:, 2*i] - projection_test[:, 2*i-1] for i in 1:Int(word_count/2)]
        arrows_x = [Bool(i%2) ? arrows[Int(i/2+0.5)][1] : 0 for i in 1:length(arrows)*2]
        arrows_y = [Bool(i%2) ? arrows[Int(i/2+0.5)][2] : 0 for i in 1:length(arrows)*2]

        @test length(arrows_x) == length(arrows_y)  # Ensure x and y lists match in size
        @test all(x -> isa(x, Number), arrows_x)  # Ensure numerical correctness
        @test all(y -> isa(y, Number), arrows_y)
    end

    @testset "Saving Plot to File" begin
        test_file = tempname() * ".png"
        show_relations("king", "queen", "man", "woman"; wv=wv, save_path=test_file)
        @test isfile(test_file)  # Ensure file was created
    end
end
