using Test
using GroupIWord2Vec

@testset "get_word2vec" begin
    # Test data setup with intuitive pattern
    words = ["cat", "dog", "bird", "fish"]  # Vocabulary of 4 test words
    embeddings = [1.0 2.0 3.0 4.0;    # First dimension of embeddings
                 5.0 6.0 7.0 8.0;    # Second dimension
                 9.0 10.0 11.0 12.0] # Third dimension
    wv = WordEmbedding(words, embeddings)
    
    @testset "basic functionality" begin
        # Test type and dimensions
        cat_vec = get_word2vec(wv, "cat")
        @test cat_vec isa Vector{Float64}
        @test length(cat_vec) == 3

        # Test with Float32 embeddings
        wv32 = WordEmbedding(words, Float32.(embeddings))
        cat_vec32 = get_word2vec(wv32, "cat")
        @test cat_vec32 isa Vector{Float32}
        @test length(cat_vec32) == 3
    end
    
    @testset "vector retrieval" begin
        # Test first word
        @test get_word2vec(wv, "cat") == [1.0, 5.0, 9.0]
        
        # Test middle word
        @test get_word2vec(wv, "bird") == [3.0, 7.0, 11.0]
        
        # Test last word
        @test get_word2vec(wv, "fish") == [4.0, 8.0, 12.0]
    end
    
    @testset "vector uniqueness" begin
        # Verify vectors are distinct
        @test get_word2vec(wv, "cat") != get_word2vec(wv, "dog")
        @test get_word2vec(wv, "dog") != get_word2vec(wv, "bird")
        @test get_word2vec(wv, "bird") != get_word2vec(wv, "fish")

        # Test immutability (ensure function returns a copy)
        cat_vec = get_word2vec(wv, "cat")
        cat_vec[1] = 999.0  # Modify retrieved vector

        # The original embeddings should remain unchanged
        @test get_word2vec(wv, "cat") == [1.0, 5.0, 9.0]  # Ensure original values are intact
    end
    
    @testset "error cases" begin
        # Test error handling
        @test_throws ArgumentError get_word2vec(wv, "unknown")
        @test_throws ArgumentError get_word2vec(wv, "")
        @test_throws ArgumentError get_word2vec(wv, " ")  # Added whitespace test
        @test_throws ArgumentError get_word2vec(wv, "Cat")  # Case sensitivity check
        @test_throws ArgumentError get_word2vec(wv, "birdd")  # Small typo test
        @test_throws ArgumentError get_word2vec(wv, " cat ")  # Leading/trailing spaces
    end
end

@testset "get_vec2word" begin
    # Test setup with clear pattern and normalized vectors
    words = ["cat", "dog", "bird", "fish"]
    embeddings = [
        1/√2  0.0   0.0   1/√2;    # First dimension
        1/√2  1.0   0.0   -1/√2;   # Second dimension
        0.0   0.0   1.0   0.0      # Third dimension
    ]
    wv = WordEmbedding(words, embeddings)
    
    @testset "exact matches" begin
        # Test with normalized vectors
        @test get_vec2word(wv, [1/√2, 1/√2, 0.0]) == "cat"     
        @test get_vec2word(wv, [0.0, 1.0, 0.0]) == "dog"       
        @test get_vec2word(wv, [0.0, 0.0, 1.0]) == "bird"      
        @test get_vec2word(wv, [1/√2, -1/√2, 0.0]) == "fish"   
    end
    
    @testset "unnormalized vectors" begin
        # Test with unnormalized vectors (same directions, different magnitudes)
        @test get_vec2word(wv, [2.0, 2.0, 0.0]) == "cat"      
        @test get_vec2word(wv, [0.0, 0.5, 0.0]) == "dog"      
        @test get_vec2word(wv, [0.0, 0.0, 3.0]) == "bird"     
    end
    
    @testset "approximate matches" begin
        # Test similar but not exact vectors
        @test get_vec2word(wv, [0.7, 0.7, 0.1]) == "cat"      
        @test get_vec2word(wv, [0.1, 0.9, 0.0]) == "dog"      
    end
    
    @testset "error cases" begin
        # Test dimension mismatch errors
        @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0])           
        @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0, 0.0, 0.0]) 
    end
end

@testset "get_any2vec" begin
    # Test setup
    words = ["cat", "dog", "bird", "fish"]
    embeddings = [1.0 2.0 3.0 4.0;
                 5.0 6.0 7.0 8.0;
                 9.0 10.0 11.0 12.0]
    wv = WordEmbedding(words, embeddings)

    @testset "word lookups" begin
        @test get_any2vec(wv, "cat") == [1.0, 5.0, 9.0]
        @test get_any2vec(wv, "dog") == [2.0, 6.0, 10.0]
    end

    @testset "vector inputs" begin
        vec = [1.0, 5.0, 9.0]
        @test get_any2vec(wv, vec) === vec  # Identity check

        wrong_dim_vec = [1.0, 5.0]  # Wrong size
        @test_throws DimensionMismatch get_any2vec(wv, wrong_dim_vec)
    end

    @testset "correct return type" begin
        wv_float32 = WordEmbedding(words, Float32.(embeddings))  # Convert to Float32 embeddings
        vec_f32 = get_any2vec(wv_float32, "cat")
        @test vec_f32 isa Vector{Float32}  # Ensures type consistency
    end

    @testset "special values" begin
        special_vec = [Inf, -Inf, NaN]
        @test get_any2vec(wv, special_vec) === special_vec  # Should return unchanged
    end

    @testset "empty vocabulary" begin
        empty_wv = WordEmbedding(String[], Matrix{Float64}(undef, 3, 0))
        @test_throws ArgumentError get_any2vec(empty_wv, "cat")  # No words exist
    end

    @testset "error cases" begin
        @test_throws ArgumentError get_any2vec(wv, "unknown_word")  # Word not in vocabulary
        @test_throws ArgumentError get_any2vec(wv, 42)  # Invalid input type
    end
end

@testset "get_vector_operation" begin
    # Test data setup with meaningful relationships
    words = ["king", "queen", "man", "woman"]
    embeddings = [1.0  2.0  3.0  4.0;    # First dimension
                 2.0  3.0  1.0  2.0;    # Second dimension
                 3.0  3.0  1.0  1.0]    # Third dimension - similar for royal pairs
    wv = WordEmbedding(words, embeddings)
    
    @testset "basic operations" begin
        # Test addition and subtraction with words and vectors
        @test get_vector_operation(wv, "king", "queen", "+") == [3.0, 5.0, 6.0]
        test_vec = [1.0, 1.0, 1.0]
        @test get_vector_operation(wv, "king", test_vec, "+") == [2.0, 3.0, 4.0]
        
        # Test relationships (king - man ≈ queen - woman)
        royal_diff = get_vector_operation(wv, "king", "man", "-")
        gender_diff = get_vector_operation(wv, "queen", "woman", "-")
        @test royal_diff ≈ gender_diff rtol=1e-5
    end
    
    @testset "similarity measures" begin
        # Test cosine similarity
        king_queen = get_vector_operation(wv, "king", "queen", "cosine")
        man_woman = get_vector_operation(wv, "man", "woman", "cosine")
        @test king_queen > 0.9  # Related pairs have high similarity
        @test isapprox(king_queen, man_woman, rtol=0.1)  # Similar relationships
        
        # Test euclidean distance
        @test get_vector_operation(wv, "king", "queen", "euclid") < 
              get_vector_operation(wv, "king", "woman", "euclid")  # Related pairs closer
    end
    
    @testset "error cases" begin
        @test_throws ArgumentError get_vector_operation(wv, "king", "queen", "invalid")
        @test_throws ArgumentError get_vector_operation(wv, "invalid", "king", "+")
        @test_throws DimensionMismatch get_vector_operation(wv, "king", [1.0], "+")
    end
end
