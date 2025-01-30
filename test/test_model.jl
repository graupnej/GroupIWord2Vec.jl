using Test
using GroupIWord2Vec

@testset "WordEmbedding" begin
    @testset "Construction & Validation" begin
        # Valid construction
        words = ["word1", "word2", "word3"]
        vectors = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 2×3 matrix
        wv = WordEmbedding(words, vectors)
        
        # Test if data is stored correctly
        @test wv.words == words
        @test wv.embeddings == vectors
        @test size(wv.embeddings, 2) == length(wv.words)
        
        # Test if dictionary is created correctly
        for (idx, word) in enumerate(words)
            @test wv.word_indices[word] == idx
        end
        
        # Test dimension mismatch
        @test_throws ArgumentError WordEmbedding(["word1", "word2"], vectors)
    end
    
    @testset "Type Parameters" begin
        # Test with different numeric types
        int_vectors = [1 2; 3 4]
        float32_vectors = Float32[1.0 2.0; 3.0 4.0]
        
        wv_int = WordEmbedding(["w1", "w2"], int_vectors)
        wv_float32 = WordEmbedding(["w1", "w2"], float32_vectors)
        
        @test eltype(wv_int.embeddings) == Int
        @test eltype(wv_float32.embeddings) == Float32
        
        # Test with different string types
        substring_words = split("word1|word2", "|")
        wv_substr = WordEmbedding(substring_words, float32_vectors)
        @test eltype(wv_substr.words) <: AbstractString
    end
end
