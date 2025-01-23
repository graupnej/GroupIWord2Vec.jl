using Test
using GroupIWord2Vec

@testset "get_vector_from_word" begin
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

@testset "get_word_from_vector" begin
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

@testset "cosine_similarity" begin
   # Test setup with normalized vectors for predictable similarities
   words = ["cat", "dog", "bird", "fish"]
   embeddings = [1.0 0.0 0.0 0.5;     # cat: [1,0,0], dog: [0,1,0], etc.
                0.0 1.0 0.0 0.5;     # Using unit vectors for easy math
                0.0 0.0 1.0 0.7071]  # fish: [0.5,0.5,0.7071] normalized
   wv = WordEmbedding(words, embeddings)
   
   @test cosine_similarity(wv, "cat", "cat") ≈ 1.0  # Same word = perfect similarity
   @test cosine_similarity(wv, "cat", "dog") ≈ 0.0  # Perpendicular vectors = no similarity
   @test cosine_similarity(wv, "cat", "fish") ≈ 0.5  # 60-degree angle = 0.5 similarity
   
   @test_throws KeyError cosine_similarity(wv, "cat", "unknown")  # First word unknown
   @test_throws KeyError cosine_similarity(wv, "unknown", "cat")  # Second word unknown
end

@testset "get_top_similarity_of_word" begin
   # Test setup: 6 words with known similarity relationships
   words = ["cat", "kitten", "puppy", "dog", "fish", "whale"]
   # 3D embeddings making: cat/kitten similar, dog/puppy similar, fish/whale similar
   embeddings = [1.0  0.9  0.2  0.1  0.0  0.0;    # First dimension
                0.0  0.1  0.8  0.9  0.1  0.1;    # Second dimension
                0.0  0.0  0.0  0.0  1.0  0.9]    # Third dimension
   wv = WordEmbedding(words, embeddings)
   
   @test get_top_similarity_of_word(wv, "cat", 1) == ["cat"]  # Self-similarity (always highest)
   @test get_top_similarity_of_word(wv, "cat", 2) == ["cat", "kitten"]  # Most similar pair
   @test get_top_similarity_of_word(wv, "fish", 3) == ["fish", "whale", "dog"]  # Top 3 similar
   @test length(get_top_similarity_of_word(wv, "cat", 6)) == 6  # Full vocabulary size
   @test_throws KeyError get_top_similarity_of_word(wv, "unknown")  # Unknown word handling
end
