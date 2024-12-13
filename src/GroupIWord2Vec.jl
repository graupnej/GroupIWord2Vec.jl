"""
Defines a module named GroupIWord2Vec which includes all functionalities
"""
module GroupIWord2Vec

# Importing modules from Julia's standard libraries
using LinearAlgebra         # For linear algebra functionalities
using DelimitedFiles        # For reading and writing delimited text files

# Exporting public types and functions for use by external modules or scripts
export Word2VecModel, load_text_model, get_word_embedding, load_fasttext_embeddings, find_similar_words

# Define a struct to represent the Word2Vec model. Containing a dictionary mapping words to indices and a matrix where each column is the embedding vector for a word
struct Word2VecModel
    vocab::Dict{String, Int}
    vectors::Matrix{Float64}
end

# Include the "functions.jl" file, which contains the implementation of functions
include("functions.jl")

end
