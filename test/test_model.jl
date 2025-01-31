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

using Test

@testset "load_embeddings" begin
    words = ["apple", "banana", "cherry"]
    embeddings = [0.1 0.2 0.3;
                  0.4 0.5 0.6;
                  0.7 0.8 0.9]

    function normalize(v) v ./ norm(v) end
    normalized_embeddings = hcat([normalize(embeddings[:, i]) for i in 1:3]...)

    mktemp() do path, io
        write(io, "3 3\n" * join([words[i] * " " * join(embeddings[:, i], " ") for i in 1:3], "\n") * "\n")
        close(io)

        wv = load_embeddings(path, format=:text, normalize_vectors=false)
        @test wv.words == words
        @test wv.embeddings ≈ embeddings
        @test load_embeddings(path, format=:text, normalize_vectors=true).embeddings ≈ normalized_embeddings
    end

    mktemp() do path, io
        write(io, "3 3\n")
        for i in 1:3
            write(io, words[i] * " ")
            write(io, reinterpret(UInt8, Float32.(embeddings[:, i])))
            write(io, "\n")
        end
        close(io)

        wv = load_embeddings(path, format=:binary, data_type=Float32, normalize_vectors=false, skip_bytes=1)
        @test wv.words == words
        @test wv.embeddings ≈ embeddings
        @test load_embeddings(path, format=:binary, data_type=Float32, normalize_vectors=true, skip_bytes=1).embeddings ≈ normalized_embeddings
    end

    mktemp() do path, io
        write(io, "3 3\n" * join([words[i] * " " * join(embeddings[:, i], " ") for i in 1:3], "\n") * "\n")
        close(io)

        @test eltype(load_embeddings(path, format=:text, data_type=Float64).embeddings) == Float64
        @test eltype(load_embeddings(path, format=:text, data_type=Float32).embeddings) == Float32
    end

    @test_throws ArgumentError load_embeddings("fake_path", format=:csv)
    @test_throws SystemError load_embeddings("non_existent_file.txt", format=:text)

    mktemp() do path, io
        close(io)
        @test_throws ArgumentError load_embeddings(path, format=:text)
    end
end
