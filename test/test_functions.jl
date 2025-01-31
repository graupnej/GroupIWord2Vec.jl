using Test
using GroupIWord2Vec

@testset "get_word2vec" begin
    # Test data setup
    words = ["cat", "dog", "bird", "fish"]
    embeddings = Float64[1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]

    wv = WordEmbedding(words, embeddings)

    @testset "basic functionality" begin
        # Test type and dimensions
        cat_vec = get_word2vec(wv, "cat")
        @test cat_vec isa Vector{Float64}
        @test length(cat_vec) == 3
    end

    @testset "vector retrieval" begin
        @test get_word2vec(wv, "cat") == [1.0, 5.0, 9.0]
        @test get_word2vec(wv, "bird") == [3.0, 7.0, 11.0]
        @test get_word2vec(wv, "fish") == [4.0, 8.0, 12.0]
    end

    @testset "vector uniqueness" begin
        @test get_word2vec(wv, "cat") != get_word2vec(wv, "dog")
        @test get_word2vec(wv, "dog") != get_word2vec(wv, "bird")
        @test get_word2vec(wv, "bird") != get_word2vec(wv, "fish")

        # Test immutability (ensure function returns a copy)
        cat_vec = get_word2vec(wv, "cat")
        cat_vec[1] = 999.0
        @test get_word2vec(wv, "cat") == [1.0, 5.0, 9.0]
    end

    @testset "error cases" begin
        # Some case sensitivity, typo etc tests
        @test_throws ArgumentError get_word2vec(wv, "unknown")
        @test_throws ArgumentError get_word2vec(wv, "")
        @test_throws ArgumentError get_word2vec(wv, " ")
        @test_throws ArgumentError get_word2vec(wv, "Cat")  
        @test_throws ArgumentError get_word2vec(wv, "birdd")
        @test_throws ArgumentError get_word2vec(wv, " cat ")
    end
end

@testset "get_vec2word" begin
    # Test setup with clear pattern and normalized vectors
    words = ["cat", "dog", "bird", "fish"]
    embeddings = [1/√2  0.0   0.0   1/√2; 1/√2  1.0   0.0   -1/√2; 0.0   0.0   1.0   0.0]
    
    wv = WordEmbedding(words, embeddings)
    
    @testset "exact matches" begin
        # Test with normalized vectors
        @test get_vec2word(wv, [1/√2, 1/√2, 0.0]) == "cat"     
        @test get_vec2word(wv, [0.0, 1.0, 0.0]) == "dog"       
        @test get_vec2word(wv, [0.0, 0.0, 1.0]) == "bird"      
        @test get_vec2word(wv, [1/√2, -1/√2, 0.0]) == "fish"   
    end
    
    @testset "unnormalized vectors" begin
        # Test with unnormalized vectors (same directions, different magnitudes)
        @test get_vec2word(wv, [2.0, 2.0, 0.0]) == "cat"      
        @test get_vec2word(wv, [0.0, 0.5, 0.0]) == "dog"      
        @test get_vec2word(wv, [0.0, 0.0, 3.0]) == "bird"     
    end
    
    @testset "approximate matches" begin
        # Test similar but not exact vectors
        @test get_vec2word(wv, [0.7, 0.7, 0.1]) == "cat"      
        @test get_vec2word(wv, [0.1, 0.9, 0.0]) == "dog"      
    end
    
    @testset "error cases" begin
        # Test dimension mismatch errors
        @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0])           
        @test_throws DimensionMismatch get_vec2word(wv, [1.0, 0.0, 0.0, 0.0]) 
    end
end

@testset "get_any2vec" begin
    words = ["cat", "dog", "bird", "fish"]
    embeddings = Float64[1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
    
    wv = WordEmbedding(words, embeddings)

    @testset "word lookups" begin
        @test get_any2vec(wv, "cat") == [1.0, 5.0, 9.0]
        @test get_any2vec(wv, "dog") == [2.0, 6.0, 10.0]
    end

    @testset "vector inputs" begin
        vec = [1.0, 5.0, 9.0]
        @test get_any2vec(wv, vec) === vec  # Identity check

        wrong_dim_vec = [1.0, 5.0]  # Wrong size
        @test_throws DimensionMismatch get_any2vec(wv, wrong_dim_vec)
    end

    @testset "special values" begin
        special_vec = [Inf, -Inf, NaN]
        @test get_any2vec(wv, special_vec) === special_vec 
    end

    @testset "empty vocabulary" begin
        empty_wv = WordEmbedding(String[], Matrix{Float64}(undef, 3, 0))
        @test_throws ArgumentError get_any2vec(empty_wv, "cat")
    end

    @testset "error cases" begin
        @test_throws ArgumentError get_any2vec(wv, "unknown_word")
    end
end

@testset "get_vector_operation" begin
    # Test data setup with meaningful relationships
    words = ["king", "queen", "man", "woman"]
    embeddings = [1.0  2.0  3.0  4.0; 2.0  3.0  1.0  2.0; 3.0  3.0  1.0  1.0]
    
    wv = WordEmbedding(words, embeddings)
    
    @testset "basic operations" begin
        # Test addition and subtraction with words and vectors
        @test isapprox(get_vector_operation(wv, "king", "queen", :+), [3.0, 5.0, 6.0], rtol=1e-5)
        test_vec = [1.0, 1.0, 1.0]
        @test isapprox(get_vector_operation(wv, "king", test_vec, :+), [2.0, 3.0, 4.0], rtol=1e-5)
        
        # Test relationships (king - man ≈ queen - woman)
        royal_diff = get_vector_operation(wv, "king", "man", :-)
        gender_diff = get_vector_operation(wv, "queen", "woman", :-)
        @test isapprox(royal_diff, gender_diff, rtol=1e-5)
    end
    
    @testset "similarity measures" begin
        # Test cosine similarity
        king_queen = get_vector_operation(wv, "king", "queen", :cosine)
        man_woman = get_vector_operation(wv, "man", "woman", :cosine)
        
        @test king_queen > 0.9  # Related pairs should have high similarity
        @test isapprox(king_queen, man_woman, rtol=0.1)  # Similar relationships should have similar cosine values

        # Check for unrelated words having lower similarity
        king_man = get_vector_operation(wv, "king", "man", :cosine)
        @test king_man < king_queen  # Less related words should have lower similarity

        # Test cosine similarity on zero vector
        zero_vec = [0.0, 0.0, 0.0]
        @test_throws ArgumentError get_vector_operation(wv, zero_vec, "queen", :cosine)

        # Test Euclidean distance
        @test get_vector_operation(wv, "king", "queen", :euclid) < get_vector_operation(wv, "king", "woman", :euclid)  # Related pairs should be closer
    end
    
    @testset "error cases" begin
        # Invalid operator
        @test_throws ArgumentError get_vector_operation(wv, "king", "queen", :invalid)
        
        # Non-existent word
        @test_throws ArgumentError get_vector_operation(wv, "invalid", "king", :+)
        
        # Mismatched dimensions
        @test_throws DimensionMismatch get_vector_operation(wv, "king", [1.0], :+)

        # Cosine similarity on zero vectors
        zero_vec = [0.0, 0.0, 0.0]
        @test_throws ArgumentError get_vector_operation(wv, zero_vec, zero_vec, :cosine)
    end
end

@testset "get_similar_words" begin
    words = ["apple", "banana", "cherry", "date", "elderberry"]
    embeddings = Float64[1.0  2.0  3.0  4.0  5.0; 2.0  3.0  4.0  5.0  6.0;3.0  4.0  5.0  6.0  7.0]

    wv = WordEmbedding(words, embeddings)

    @testset "basic functionality" begin
        # Test that the function returns a vector of words
        similar_words = get_similar_words(wv, "banana", 3)
        @test similar_words isa Vector{String}
        @test length(similar_words) == 3
    end

    @testset "consistency of string and vector inputs" begin
        # Retrieve vector for "banana"
        banana_vec = get_any2vec(wv, "banana")
        similar_words_from_vec = get_similar_words(wv, banana_vec, 3)

        # The results should be the same
        @test get_similar_words(wv, "banana", 3) == similar_words_from_vec
    end

    @testset "similarity ranking order" begin
        # Using "cherry" as query
        query_word = "cherry"
        query_vec = get_any2vec(wv, query_word)
        
        # Compute expected ranking based on cosine similarity
        similarities = wv.embeddings' * query_vec
        sorted_indices = sortperm(similarities, rev=true)[1:3]
        expected_words = wv.words[sorted_indices]

        retrieved_words = get_similar_words(wv, query_word, 3)

        @test retrieved_words == expected_words  # Ensure correct ranking
    end

    @testset "error cases" begin
        @test_throws ArgumentError get_similar_words(wv, "mango", 3)
        
        # Empty string or whitespace should throw an error
        @test_throws ArgumentError get_similar_words(wv, "", 3)
        @test_throws ArgumentError get_similar_words(wv, " ", 3)

        # Word with incorrect capitalization should throw an error (if case-sensitive)
        @test_throws ArgumentError get_similar_words(wv, "Banana", 3)
    end
end

@testset "get_word_analogy" begin
    # Test data setup
    words = ["king", "queen", "man", "woman", "prince", "princess"]
    embeddings = Float64[
        1.0  0.9  0.8  0.7  0.6  0.5;
        0.8  0.7  0.6  0.5  0.4  0.3;
        0.6  0.5  0.4  0.3  0.2  0.1
    ]  # 3D vectors representing semantic relationships

    wv = WordEmbedding(words, embeddings)

    @testset "basic functionality" begin
        result = get_word_analogy(wv, "king", "man", "woman", 1)
        @test result == ["queen"]
    end

    @testset "vector and word inputs consistency" begin
        king_vec, man_vec, woman_vec = get_any2vec(wv, "king"), get_any2vec(wv, "man"), get_any2vec(wv, "woman")
        @test get_word_analogy(wv, "king", "man", "woman", 1) == get_word_analogy(wv, king_vec, man_vec, woman_vec, 1)
    end

    @testset "exclusion of input words" begin
        result = get_word_analogy(wv, "king", "man", "woman", 3)
        exclude_set = Set(["king", "man", "woman"])
        
        # Ensure that excluded words appear at most once (handling edge cases where n > valid candidates)
        @test count(word -> word in exclude_set, result) == 0
    end

    @testset "error cases" begin
        @test_throws ArgumentError get_word_analogy(wv, "king", "man", "woman", 0)
        @test_throws ArgumentError get_word_analogy(wv, "king", "man", "woman", 3, metric=:invalid_metric)
        @test_throws ArgumentError get_word_analogy(wv, "unknown", "man", "woman", 3)
    end
end
