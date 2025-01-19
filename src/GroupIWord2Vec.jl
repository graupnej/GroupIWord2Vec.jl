"""
Defines a module named GroupIWord2Vec which includes all functionalities
"""
module GroupIWord2Vec

# Importing modules from Julia's standard libraries
using LinearAlgebra         # For linear algebra functionalities
using DelimitedFiles        # For reading and writing delimited text files
import Word2Vec_jll

# Exporting public types and functions for use by external modules or scripts
export word2vec, get_vector, WordEmbedding, load_embeddings, read_binary_format, read_text_format

# Define the mutable struct for word embeddings
mutable struct WordEmbedding{S<:AbstractString, T<:Real}
    words::Vector{S}
    embeddings::Matrix{T}
    word_indices::Dict{S, Int}
    
    # Custom constructor with validation
    function WordEmbedding(words::Vector{S}, matrix::Matrix{T}) where {S<:AbstractString, T<:Real}
        if length(words) != size(matrix, 2)
            throw(ArgumentError("Number of words ($(length(words))) must match matrix columns ($(size(matrix, 2)))"))
        end
        indices = Dict(word => idx for (idx, word) in enumerate(words))
        new{S,T}(words, matrix, indices)
    end
end

# Include the "functions.jl" file, which contains the implementation of functions
include("functions.jl")

end
