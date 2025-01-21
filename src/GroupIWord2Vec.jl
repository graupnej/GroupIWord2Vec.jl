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
      get_similarity,        # Function to find top n similar words
      plot_similarity,       # Function to visualize similarities
      cosine_similarity     # Function to compute similarity of two words

# Include all the functions defined in functions.jl
include("functions.jl")

end
