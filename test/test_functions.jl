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
    # Test setup with clear pattern and sufficient vocabulary
    words = ["cat", "dog", "bird", "fish"]
    # Each column represents a word vector with clear numerical pattern
    embeddings = [1.0 2.0 3.0 4.0;     # First dimension
                 5.0 6.0 7.0 8.0;     # Second dimension
                 9.0 10.0 11.0 12.0]  # Third dimension
    wv = WordEmbedding(words, embeddings)
    
    # Test exact vector matches for all words
    @test get_vec2word(wv, [1.0, 5.0, 9.0]) == "cat"    # First word
    @test get_vec2word(wv, [2.0, 6.0, 10.0]) == "dog"   # Second word
    @test get_vec2word(wv, [3.0, 7.0, 11.0]) == "bird"  # Third word
    @test get_vec2word(wv, [4.0, 8.0, 12.0]) == "fish"  # Fourth word
    
    # Test scale invariance (cosine similarity property)
    @test get_vec2word(wv, 2 .* [1.0, 5.0, 9.0]) == "cat"    # Scaled first vector
    @test get_vec2word(wv, 0.5 .* [4.0, 8.0, 12.0]) == "fish" # Scaled last vector
    
    # Test similar but not exact vectors (nearest neighbor)
    @test get_vec2word(wv, [1.1, 5.1, 9.1]) == "cat"    # Close to first word
    @test get_vec2word(wv, [3.9, 7.9, 11.9]) == "fish"  # Close to last word
    
    # Test dimension mismatch errors
    @test_throws DimensionMismatch get_vec2word(wv, [1.0, 5.0])  # Too short
    @test_throws DimensionMismatch get_vec2word(wv, [1.0, 5.0, 9.0, 13.0])  # Too long
end
