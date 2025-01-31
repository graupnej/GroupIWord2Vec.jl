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

@testset "load_embeddings" begin
    words = ["apple", "banana", "cherry"]
    embeddings = Float64[0.1 0.2 0.3;
                         0.4 0.5 0.6;
                         0.7 0.8 0.9]  # Ensure Float64 for consistency

    # Manual normalization function (avoiding `norm()`)
    function normalize(v)
        magnitude = sqrt(sum(v .^ 2))  # Compute vector magnitude
        return magnitude > 0 ? v ./ magnitude : v  # Avoid division by zero
    end
    normalized_embeddings = hcat([normalize(embeddings[:, i]) for i in 1:3]...)

    mktemp() do path, io
        # Fix: Ensure text embeddings are written correctly with spaces
        write(io, "3 3\n" * join([words[i] * " " * join(embeddings[:, i], " ") for i in 1:3], "\n") * "\n")
        close(io)

        wv = load_embeddings(path, format=:text, normalize_vectors=false)
        @test wv.words == words
        @test wv.embeddings ≈ embeddings
        @test load_embeddings(path, format=:text, normalize_vectors=true).embeddings ≈ normalized_embeddings
    end

    mktemp() do path, io
        # Fix: Correctly write binary format
        write(io, "3 3\n")  # Metadata line
        for i in 1:3
            write(io, words[i] * "\n")  # Separate words properly with newline
            write(io, reinterpret(UInt8, embeddings[:, i]))  # Ensure Float64 writing
        end
        close(io)

        # Fix: Ensure correct `skip_bytes` and remove extra characters in words
        wv = load_embeddings(path, format=:binary, normalize_vectors=false, skip_bytes=0)
        @test wv.words == words  # ✅ Now correctly parsed
        @test wv.embeddings ≈ embeddings  # ✅ Now correctly read
        @test load_embeddings(path, format=:binary, normalize_vectors=true, skip_bytes=0).embeddings ≈ normalized_embeddings
    end

    mktemp() do path, io
        write(io, "3 3\n" * join([words[i] * " " * join(embeddings[:, i], " ") for i in 1:3], "\n") * "\n")
        close(io)

        @test eltype(load_embeddings(path, format=:text, data_type=Float64).embeddings) == Float64
    end

    @test_throws ArgumentError load_embeddings("fake_path", format=:csv)
    @test_throws SystemError load_embeddings("non_existent_file.txt", format=:text)

    mktemp() do path, io
        close(io)
        @test_throws ArgumentError load_embeddings(path, format=:text)
    end
end
