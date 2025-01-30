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
    end
    
    @testset "error cases" begin
        # Test error handling
        @test_throws ArgumentError get_word2vec(wv, "unknown")
        @test_throws ArgumentError get_word2vec(wv, "")
        @test_throws ArgumentError get_word2vec(wv, " ")  # Added whitespace test
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
    # Test data setup with intuitive pattern
    words = ["cat", "dog", "bird", "fish"]  # Vocabulary of 4 test words
    embeddings = [1.0 2.0 3.0 4.0;    # First dimension of embeddings
                 5.0 6.0 7.0 8.0;    # Second dimension
                 9.0 10.0 11.0 12.0] # Third dimension
    wv = WordEmbedding(words, embeddings)
    
    @testset "word lookups" begin
        # Test vectors and types for multiple positions
        cat_vec = get_any2vec(wv, "cat")
        @test cat_vec == [1.0, 5.0, 9.0] && cat_vec isa Vector{Float64}
        @test get_any2vec(wv, "bird") == [3.0, 7.0, 11.0]  # Middle word
        @test get_any2vec(wv, "fish") == [4.0, 8.0, 12.0]  # Last word
        @test get_any2vec(wv, "cat") != get_any2vec(wv, "dog")  # Distinctness
    end
    
    @testset "vector inputs" begin
        # Test vector identity and special cases
        vec1 = [1.0, 5.0, 9.0]
        @test get_any2vec(wv, vec1) === vec1  # Identity preservation
        @test get_any2vec(wv, zeros(3)) === zeros(3)  # Zero vector
        @test get_any2vec(wv, [1e6, 1e6, 1e6]) isa Vector{Float64}  # Large values
        @test get_any2vec(wv, [Inf, -Inf, NaN]) isa Vector{Float64}  # Special values
    end
    
    @testset "error cases" begin
        # Test various error conditions
        @test_throws ArgumentError get_any2vec(wv, "nonexistent_word")
        @test_throws ArgumentError get_any2vec(wv, "")  # Empty string
        @test_throws ArgumentError get_any2vec(wv, 42)  # Wrong type
        @test_throws ArgumentError get_any2vec(wv, [1, 2, 3])  # Integer vector
        @test_throws DimensionMismatch get_any2vec(wv, Float64[])  # Empty vector
        @test_throws DimensionMismatch get_any2vec(wv, [1.0, 2.0])  # Wrong size
    end
end
