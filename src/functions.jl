"""
The load_text_model function takes a file in text format containing the pre-trained Word2Vec model as input 
and gives an object of type Word2VecModel as output. This object includes a dictionary mapping words
to their indices and a matrix of vectors where each column corresponds to a word's vector
"""

"""
     word2vec(train, output; size=100, window=5, sample=1e-3, hs=0,  negative=5, threads=12, iter=5, min_count=5, alpha=0.025, debug=2, binary=1, cbow=1, save_vocal=Nothing(), read_vocab=Nothing(), verbose=false,)

    Parameters for training:
        train <file>
            Use text data from <file> to train the model
        output <file>
            Use <file> to save the resulting word vectors / word clusters
        size <Int>
            Set size of word vectors; default is 100
        window <Int>
            Set max skip length between words; default is 5
        sample <AbstractFloat>
            Set threshold for occurrence of words. Those that appear with
            higher frequency in the training data will be randomly
            down-sampled; default is 1e-5.
        hs <Int>
            Use Hierarchical Softmax; default is 1 (0 = not used)
        negative <Int>
            Number of negative examples; default is 0, common values are 
            5 - 10 (0 = not used)
        threads <Int>
            Use <Int> threads (default 12)
        iter <Int>
            Run more training iterations (default 5)
        min_count <Int>
            This will discard words that appear less than <Int> times; default
            is 5
        alpha <AbstractFloat>
            Set the starting learning rate; default is 0.025
        debug <Int>
            Set the debug mode (default = 2 = more info during training)
        binary <Int>
            Save the resulting vectors in binary moded; default is 0 (off)
        cbow <Int>
            Use the continuous back of words model; default is 1 (skip-gram
            model)
        save_vocab <file>
            The vocabulary will be saved to <file>
        read_vocab <file>
            The vocabulary will be read from <file>, not constructed from the
            training data
        verbose <Bool>
            Print output from training 
"""
function train_model(train::AbstractString, output::AbstractString; 
                  size::Int=100, window::Int=5, sample::AbstractFloat=1e-3,
                  hs::Int=0, negative::Int=5, threads::Int=12, iter::Int=5, 
                  min_count::Int=5, alpha::AbstractFloat=0.025,
                  debug::Int=2, binary::Int=0, cbow::Int=1, 
                  save_vocab=Nothing(), read_vocab=Nothing(),
                  verbose::Bool=false)

    parameters = AbstractString[]
    args = ["-train", "-output", "-size", "-window", "-sample", "-hs",
            "-negative", "-threads", "-iter", "-min-count", "-alpha", 
            "-debug", "-binary", "-cbow"]
    values = [train, output, size, window, sample, hs, negative, threads,
              iter, min_count, alpha, debug, binary, cbow]

    for (arg, value) in zip(args, values)
        push!(parameters, arg)
        push!(parameters, string(value))
    end
    if save_vocab != Nothing()
        push!(parameters, "-save-vocab")
        push!(parameters, string(save_vocab))
    end
    if read_vocab != Nothing()
        push!(parameters, "-read-vocab")
        push!(parameters, string(read_vocab))
    end        
    Word2Vec_jll.word2vec() do command
        run(`$(command) $(parameters)`)
    end
end

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

"""
# This is the main function that loads word embeddings from a file. It supports different formats
# and allows customization of how the data should be loaded. It requires the following parameters:
#
# 1) path: Where to find the file
# 2) format: What format the file is in (:text or :binary) with default text
# 3) data_type: What type of numbers to use (defaults to Float64)
# 4) normalize_vectors: Whether to normalize the vectors (defaults to true)
# 5) separator: What character separates values (defaults to space)
# 6) skip_bytes: How many bytes to skip in binary format (defaults to 1)
#
# We use two types of formats because
# Text Format
# - is easy to inspect because human readable
# - can be edited manually
# - but takes more storage space and is slower to read
#
# Binary Format
# - not human readable because stored as raw bytes
# - way smaller file size and fast to read
# - complexer to debug
#
# Word2Vec models often come in both formats because the text format is for inspection and modifcation
# and binary format is for efficient application usage
"""
# Main loading function
function load_embeddings(path::AbstractString; format::Symbol=:text,data_type::Type{T}=Float64,normalize_vectors::Bool=true,separator::Char=' ',skip_bytes::Int=1) where T<:Real
     # For a text file use the read_text_format function
    if format == :text
        return read_text_format(path, data_type, normalize_vectors, separator)
     # For a binary file use the read_binary_format function
    elseif format == :binary
        return read_binary_format(path, data_type, normalize_vectors, separator, skip_bytes)
    else
        throw(ArgumentError("Unsupported format: $format"))
    end
end

"""
# This function reads word embeddings (word->vector mappings) from a text file
# It requires the following Parameters:
#   filepath: where the file is located
#   T: what kind of numbers we want (like decimal numbers)
#   normalize: whether to make all vectors have length 1
#               ---> This can be useful for comparison since the length of the vector does not
#                    matter, only its direction
#   separator: what character separates the values in the file (like space or comma)
"""
function read_text_format(filepath::AbstractString, ::Type{T},normalize::Bool,separator::Char) where T<:Real
    open(filepath) do file
          # Read header with vocabulary size and vector dimension
          header = split(strip(readline(file)), separator)
          vocab_size, vector_size = parse.(Int, header)

          # Prepare arrays for words and vectors
          words = Vector{String}(undef, vocab_size)
          vectors = Matrix{T}(undef, vector_size, vocab_size)

          # For each remaining line in the file:
          for (idx, line) in enumerate(eachline(file))
            # Split the line into parts using our separator
            parts = split(strip(line), separator)
          
            # The first part is the word
            words[idx] = parts[1]
          
            # The rest are numbers that make up the vector
            vector = parse.(T, parts[2:end])
          
            # If normalize is true, make the vector length 1
            if normalize
                vector = vector ./ norm(vector)
            end
          
            # Store the vector in our matrix
            vectors[:, idx] = vector
          end
          
          # Create a WordEmbedding object with our words and vectors
          return WordEmbedding(words, vectors)
    end
end

"""
# This function reads word embeddings (word->vector mappings) from a binary file
# It requires the following Parameters:
#   filepath: where the file is located
#   T: what kind of numbers we want (like decimal numbers)
#   normalize: whether to make all vectors have length 1
#               ---> This can be useful for comparison since the length of the vector does not
#                    matter, only its direction
#   separator: what character separates the values in the file (like space or comma)
#   skip_bytes: how many bytes to skip after each word-vector pair (usually for handling separators)
# Instead of reading lines of text and parsing numbers it reads words until it hits a separator
# Reads raw bytes and converts them directly to numbers
"""
function read_binary_format(filepath::AbstractString,::Type{T},normalize::Bool,separator::Char,skip_bytes::Int) where T<:Real

    open(filepath, "r") do file
        # Read header with vocabulary size and vector dimension
        header = split(strip(readline(file)), separator)
        vocab_size, vector_size = parse.(Int, header)

          # Prepare arrays for words and vectors
        words = Vector{String}(undef, vocab_size)
        vectors = zeros(T, vector_size, vocab_size)

        # Calculate how many bytes each vector takes. Each number in the vector is stored as a Float32
        vector_bytes = sizeof(Float32) * vector_size

        for i in 1:vocab_size
            # Read the word until we hit the separator
            words[i] = strip(readuntil(file, separator))

            # Read the raw bytes for the vector and interpret them as Float32 numbers (faster than parsing text numbers)
            vector = reinterpret(Float32, read(file, vector_bytes))

            # Normalize if requested
            if normalize
                vector = vector ./ norm(vector)
            end

            # Convert to desired number type and store
            vectors[:, i] = T.(vector)
                    
            # Skip extra bytes (like newlines) after each word-vector pair
            read(file, skip_bytes)
        end

        # Return the WordEmbedding object
        return WordEmbedding(words, vectors)
    end
end

"""
# Purpose: Get the vector representation of a specific word from the WordEmbedding
"""
function get_vector(wv::WordEmbedding, word)
    # Performance steps are separated using semicolon
    # Step 1: idx = wv.word_indices[word]     --> Checks the word's index in the dictionary
    # Step 2: wv.embeddings[:,idx]            --> Checks column from the embeddings matrix
    idx = wv.word_indices[word]; wv.embeddings[:,idx]
end

"""
Return the cosine similarity value between two words `word1` and `word2`.
"""
function cosine_similarity(wv::WordEmbedding, word_1, word_2)
   # 1. Get vector representations for both words 
   # 2. Transpose first vector (') and multiply (*) with second vector
   # 3. Since word vectors are normalized, this dot product directly gives the cosine similarity
   return get_vector(wv, word_1)'*get_vector(wv, word_2)
end

"""
# Purpose: Find the n (default n = 10) most similar words to a given word
"""
function get_similarity(wv::WordEmbedding, word, n=10)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*get_vector(wv, word)

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]

    # Step 3: Get the similarity scores for these positions
    topn_metrics = metrics[topn_positions]

    # Return both positions and their similarity scores
    return topn_positions, topn_metrics
end
