using Test
using GroupIWord2Vec

@testset "get_vector_from_word" begin
   vocab = ["cat", "dog", "bird", "fish"]
   embedding_dim = 3
   embeddings = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
   word_indices = Dict(word => i for (i, word) in enumerate(vocab))
   wv = WordEmbedding(embeddings, word_indices)
   
   @test get_vector_from_word(wv, "cat") == [1.0, 5.0, 9.0]
   @test get_vector_from_word(wv, "fish") == [4.0, 8.0, 12.0]
   @test_throws KeyError get_vector_from_word(wv, "unknown")
   @test_throws KeyError get_vector_from_word(wv, "")
end
