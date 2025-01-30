using Test
using GroupIWord2Vec

@testset "get_word2vec" begin
   # Test data setup
   words = ["cat", "dog", "bird", "fish"]  # Vocabulary of 4 test words
   # 3x4 embedding matrix: each column is a word vector
   embeddings = [1.0 2.0 3.0 4.0;    # First dimension of embeddings
                5.0 6.0 7.0 8.0;    # Second dimension
                9.0 10.0 11.0 12.0] # Third dimension
   wv = WordEmbedding(words, embeddings)  # Create embedding object
   
   # Test vector retrieval for first word "cat" (first column)
   @test get_vector_from_word(wv, "cat") == [1.0, 5.0, 9.0]
   
   # Test vector retrieval for last word "fish" (fourth column)
   @test get_vector_from_word(wv, "fish") == [4.0, 8.0, 12.0]
   
   # Test error handling for non-existent words
   @test_throws KeyError get_vector_from_word(wv, "unknown")
   @test_throws KeyError get_vector_from_word(wv, "")
end

@testset "get_vec2word" begin
   # Test setup with 4 words and 3D embeddings
   words = ["cat", "dog", "bird", "fish"]
   embeddings = [1.0 2.0 3.0 4.0;     # Each column represents a word vector
                5.0 6.0 7.0 8.0;     # [1,5,9] = cat, [2,6,10] = dog, etc.
                9.0 10.0 11.0 12.0]
   wv = WordEmbedding(words, embeddings)
   
   # Test vector->word retrieval for all words
   @test get_word_from_vector(wv, [1.0, 5.0, 9.0]) == "cat"    # First word
   @test get_word_from_vector(wv, [2.0, 6.0, 10.0]) == "dog"   # Second word
   @test get_word_from_vector(wv, [3.0, 7.0, 11.0]) == "bird"  # Third word
   @test get_word_from_vector(wv, [4.0, 8.0, 12.0]) == "fish"  # Fourth word
   
   # Test non-existent vector error
   @test_throws ArgumentError get_word_from_vector(wv, [0.0, 0.0, 0.0])
end

@testset "get_any2vec" begin
    # Create test embedding with meaningful test data
    vocab = ["dog", "cat", "house"]
    vectors = [
        1.0 2.0 3.0;
        -1.0 -2.0 1.5;
        0.5 0.7 -1.0
    ]'
    wv = WordEmbedding(vocab, vectors)
    
    # Test string input
    dog_vec = get_any2vec(wv, "dog")
    @test dog_vec isa Vector{Float64}
    @test dog_vec == vectors[1, :]
    @test length(dog_vec) == size(vectors, 1)
    
    # Test vector input
    test_vec = [1.0, 2.0, 3.0]
    @test get_any2vec(wv, test_vec) === test_vec
    
    # Test error cases
    @test_throws KeyError get_any2vec(wv, "nonexistent_word")
    @test_throws ArgumentError get_any2vec(wv, 42)
    @test_throws DimensionMismatch get_any2vec(wv, [1.0, 2.0])  # Wrong size vector
end
