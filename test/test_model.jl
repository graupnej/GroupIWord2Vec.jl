using Test
using GroupIWord2Vec

@testset "WordEmbedding" begin
    @testset "Construction & Validation" begin
        # Valid construction
        words = ["word1", "word2", "word3"]
        vectors = Float64[1.0 2.0 3.0; 4.0 5.0 6.0]  # Ensure it's Float64
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
        # Convert Int64 and Float32 matrices to Float64
        int_vectors = Float64[1 2; 3 4]  # Convert to Float64
        float32_vectors = Float64[1.0 2.0; 3.0 4.0]  # Convert to Float64
        
        wv_int = WordEmbedding(["w1", "w2"], int_vectors)  # ✅ Now valid
        wv_float32 = WordEmbedding(["w1", "w2"], float32_vectors)  # ✅ Now valid

        @test eltype(wv_int.embeddings) == Float64
        @test eltype(wv_float32.embeddings) == Float64
        
        # Fix for SubString issue
        substring_words = collect(String, split("word1|word2", "|"))  # Convert to Vector{String}
        wv_substr = WordEmbedding(substring_words, float32_vectors)
        @test eltype(wv_substr.words) == String  # ✅ Now correct
    end
end
