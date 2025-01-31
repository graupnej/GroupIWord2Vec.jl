module GroupIWord2Vec
"""
# GroupIWord2Vec
This is the main module file that organizes all the word embedding functionality.

## Types
- `WordEmbedding`: Main data structure for word embeddings

## Main functions
- `train_model`:             Train new word embeddings
- `load_embeddings`:         Load pre-trained embeddings
- `get_word2vec`:            Function to get a word's embedded vector
- `get_vec2word`:            Function to get a vector's word
- `get_any2vec`:             Function to handle word/ vector input and convert input word/ vector into vector
- `get_similar_words`:       Function to find top n similar words as strings
- `get_vector_operation`:    Function to find perform operation on 2 input words/vectors: sum, subtract, dot-product, euclidean distance
- `get_word_analogy`:        Function to use vector math to compute analogous words
- `reduce_to_2d`:            Function performing Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset
- `show_relations`:          Function to visualise vectors and vector distances in 2D
- `create_vocabulary`:       Creates a custom vocabulary
- `create_custom_model`:            Creates a trainable model with Flux
- `train_custom_model`:             Trains the model
- `save_custom_model`:       Saves the model into a txt
"""

# Include all the functions defined in functions.jl, model.jl and show_relations.jl
include("model.jl")
include("functions.jl")
include("show_relations.jl")

# Importing modules from Julia's standard libraries
using LinearAlgebra         # Provides functions for vector/matrix operations (e.g. multiplication and normalization)
using DelimitedFiles        # Helps read/write files with separators (like binary files)
using Statistics            # For basic statistical operations (mean, std, var, etc.)
using Plots                 # For visualization functions
using Flux, ProgressMeter
using Flux: train!
using Random
using OneHotArrays
using Statistics
using BenchmarkTools
import Word2Vec_jll         # Links to the underlying Word2Vec implementation (C code)

# Make these functions and types available to anyone using this module. Other programs can use these functions after importing GroupIWord2Vec
export      train_model,           
            WordEmbedding,          
            load_embeddings,        
            read_binary_format,     
            read_text_format,       
            get_word2vec,           
            get_vec2word,           
            get_any2vec,            
            get_similar_words,      
            get_vector_operation,   
            get_word_analogy,       
            reduce_to_2d,
            show_relations,
            create_vocabulary,
            create_custom_model,
            train_custom_model,
            save_custom_model
end
