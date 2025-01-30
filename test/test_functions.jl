using Test
using GroupIWord2Vec

@testset "get_word2vec" begin
    # Test data setup with intuitive pattern
    words = ["cat", "dog", "bird", "fish"]  # Vocabulary of 4 test words
    # 3x4 embedding matrix: each column is a word vector
    embeddings = [1.0 2.0 3.0 4.0;    # First dimension of embeddings
                 5.0 6.0 7.0 8.0;    # Second dimension
                 9.0 10.0 11.0 12.0] # Third dimension
    wv = WordEmbedding(words, embeddings)  # Create embedding object
    
    # Test type and content for first word
    cat_vec = get_word2vec(wv, "cat")
    @test cat_vec isa Vector{Float64}
    @test cat_vec == [1.0, 5.0, 9.0]
    @test length(cat_vec) == 3
    
    # Test vector retrieval for middle and last words
    @test get_word2vec(wv, "bird") == [3.0, 7.0, 11.0]
    @test get_word2vec(wv, "fish") == [4.0, 8.0, 12.0]
    
    # Verify vectors are distinct
    @test get_word2vec(wv, "cat") != get_word2vec(wv, "dog")
    
    # Test error handling
    @test_throws ArgumentError get_word2vec(wv, "unknown")
    @test_throws ArgumentError get_word2vec(wv, "")
end

@testset "get_vec2word" begin
    # Test setup with clear pattern and normalized vectors
    words = ["cat", "dog", "bird", "fish"]
    # Each column is a normalized vector 
    embeddings = [
        1/√2  0.0   0.0   1/√2;    # First dimension
        1/√2  1.0   0.0   -1/√2;   # Second dimension
        0.0   0.0   1.0   0.0      # Third dimension
    ]
    wv = WordEmbedding(words, embeddings)
    
    # Test with normalized vectors
    @test get_vec2word(wv, [1/√2, 1/√2, 0.0]) == "cat"     # Matches first vector
    @test get_vec2word(wv, [0.0, 1.0, 0.0]) == "dog"       # Matches second vector
    @test get_vec2word(wv, [0.0, 0.0, 1.0]) == "bird"      # Matches third vector
    @test get_vec2word(wv, [1/√2, -1/√2, 0.0]) == "fish"   # Matches fourth vector
    
    # Test with unnormalized vectors (same directions, different magnitudes)
    @test get_vec2word(wv, [2.0, 2.0, 0.0]) == "cat"       # Scaled first vector
    @test get_vec2word(wv, [0.0, 0.5, 0.0]) == "dog"       # Scaled second vector
    @test get_vec2word(wv, [0.0, 0.0, 3.0]) == "bird"      # Scaled third vector
    
    # Test similar but not exact vectors
    @test get_vec2word(wv, [0.7, 0.7, 0.1]) == "cat"       # Closer to first vector
    @test get_vec2word(wv, [0.1, 0.9, 0.0]) == "dog"       # Closer to second vector
    
    # Test dimension mismatch errors
    @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0])              # Too short
    @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0, 0.0, 0.0])    # Too long
end
