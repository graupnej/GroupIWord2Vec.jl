using Test
using GroupIWord2Vec

mutable struct WordEmbedding{S<:AbstractString, T<:Real}
     # List of all words in the vocabulary
    words::Vector{S}

     # Matrix containing all word vectors
     # - Each column represents one word's vector
     # - If we have 3 words and vectors of length 4 we have a 4×3 matrix
    embeddings::Matrix{T}

     # Dictionary for quick word lookup
     # Maps each word to its position in the words vector and embeddings matrix
    word_indices::Dict{S, Int}
    
     # It makes sure the data is valid and sets everything up correctly
    function WordEmbedding(words::Vector{S}, matrix::Matrix{T}) where {S<:AbstractString, T<:Real}
          # Check if the number of words matches the number of vectors (3 words need 3 vectors)
        if length(words) != size(matrix, 2)
            throw(ArgumentError("Number of words ($(length(words))) must match matrix columns ($(size(matrix, 2)))"))
        end
          
          # This makes a dictionary where each word points to its position
        indices = Dict(word => idx for (idx, word) in enumerate(words))

          # Create the new WordEmbedding with all its parts
        new{S,T}(words, matrix, indices)
    end
end

@testset "get_vector_from_word" begin
    words = AbstractString["cat", "dog", "bird", "fish"]
    embeddings = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]
    wv = WordEmbedding(words, embeddings)
    
    @test get_vector_from_word(wv, "cat") == [1.0, 5.0, 9.0]
    @test get_vector_from_word(wv, "fish") == [4.0, 8.0, 12.0]
    @test_throws KeyError get_vector_from_word(wv, "unknown")
    @test_throws KeyError get_vector_from_word(wv, "")
end

