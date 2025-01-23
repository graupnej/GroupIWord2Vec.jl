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

