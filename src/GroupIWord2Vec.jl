"""
# GroupIWord2Vec

This is the main module file that organizes all the word embedding functionality.

## Types
- `WordEmbedding`: Main data structure for word embeddings

## Functions
- `train_model`: Train new word embeddings
- `load_embeddings`: Load pre-trained embeddings
- `get_vector`: Get a word's vector
- `get_similarity`: Find top n similar words
- `cosine_similarity`: Compute similarity of two words
"""
module GroupIWord2Vec

# Include all the functions defined in functions.jl and model.jl
include("model.jl")
include("functions.jl")
include("show_relations.jl")

# Importing modules from Julia's standard libraries
using LinearAlgebra         # Provides functions for vector/matrix operations (e.g. multiplication and normalization)
using DelimitedFiles        # Helps read/write files with separators (like binary files)
using Statistics
using Plots
import Word2Vec_jll         # Links to the underlying Word2Vec implementation (C code)

# Make these functions and types available to anyone using this module. Other programs can use these functions after importing GroupIWord2Vec
export train_model,          # Function to train new word embeddings
      WordEmbedding,         # The main data structure for word embeddings
      load_embeddings,       # Function to load pre-trained embeddings
      read_binary_format,    # Function to read binary embedding files
      read_text_format,      # Function to read text embedding files
      get_vector_from_word,            # Function to get a word's vector
      get_word_from_vector,
      cosine_similarity,     # Function to compute similarity of two words
      get_top_similarity_of_word, # Function to find top n similar words as strings
      get_top_similarity_of_vector,
      word_analogy,
      word_addition,
      word_subtraction,
      euclidean_distance,
      show_relations

end
