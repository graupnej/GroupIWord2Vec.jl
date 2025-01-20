"""
# This is the main module file that organizes all the word embedding functionality
"""
module GroupIWord2Vec

# Importing modules from Julia's standard libraries
using LinearAlgebra         # Provides functions for vector/matrix operations (e.g. multiplication and normalization)
using DelimitedFiles        # Helps read/write files with separators (like binary files)
# using Plots                 # For visualization functions
import Word2Vec_jll         # Links to the underlying Word2Vec implementation (C code)

# Make these functions and types available to anyone using this module. Other programs can use these functions after importing GroupIWord2Vec
export train_model,          # Function to train new word embeddings
      WordEmbedding,         # The main data structure for word embeddings
      load_embeddings,       # Function to load pre-trained embeddings
      read_binary_format,    # Function to read binary embedding files
      read_text_format,      # Function to read text embedding files
      get_vector,            # Function to get a word's vector
      get_similarity,        # Function to find similar words
      plot_similarity        # Function to visualize similarities


"""
# WordEmbedding is a structure that holds three related pieces of information
# 1) A list of all words
# 2) The corresponding vectors
# 3) A lookup dictionary
# to keep words and their vectors organized and linked together
# It is mutable (can be modified after creation) and works with any string and number types
"""

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

# Include all the functions defined in functions.jl
include("functions.jl")

end
