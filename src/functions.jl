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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# WordEmbedding is a structure that holds three related pieces of information
# 1) A list of all words
# 2) The corresponding vectors
# 3) A lookup dictionary
# to keep words and their vectors organized and linked together
# It is mutable (can be modified after creation) and works with any string and number types
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
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
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# This function reads word embeddings (word->vector mappings) from a text file
# It requires the following Parameters:
#   filepath: where the file is located
#   T: what kind of numbers we want (like decimal numbers)
#   normalize: whether to make all vectors have length 1
#               ---> This can be useful for comparison since the length of the vector does not
#                    matter, only its direction
#   separator: what character separates the values in the file (like space or comma)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
function read_text_format(filepath::AbstractString, ::Type{T},normalize::Bool,separator::Char) where T<:Real

    open(filepath) do file
          
        # Read the first line which contains information on two numbers: 
        # how many words (vocab_size) and how long each vector is (vector_size)
        header = split(strip(readline(file)), separator)
        vocab_size, vector_size = parse.(Int, header
               
        # Create empty arrays to store our data:
        # - words: will hold all the words
        # - vectors: will hold all the corresponding number vectors
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

# Binary format reader
function read_binary_format(filepath::AbstractString,::Type{T},normalize::Bool,separator::Char,skip_bytes::Int) where T<:Real

    open(filepath, "r") do file
        # Parse header
        header = split(strip(readline(file)), separator)
        vocab_size, vector_size = parse.(Int, header)

        words = Vector{String}(undef, vocab_size)
        vectors = zeros(T, vector_size, vocab_size)
        vector_bytes = sizeof(Float32) * vector_size

        for i in 1:vocab_size
            words[i] = strip(readuntil(file, separator))
            vector = reinterpret(Float32, read(file, vector_bytes))

            if normalize
                vector = vector ./ norm(vector)
            end

            vectors[:, i] = T.(vector)
            read(file, skip_bytes)  # handle newline or other separators
        end

        return WordEmbedding(words, vectors)
    end
end

"""
     get_vector(wv, word)

# Return the vector representation of `word` from the WordVectors `wv`.
"""
get_vector(wv::WordEmbedding, word) = (idx = wv.word_indices[word]; wv.embeddings[:,idx])

"""
    cosine(wv, word, n=10)

Return the position of `n` (by default `n = 10`) neighbors of `word` and their
cosine similarities.
"""
function get_similarity(wv::WordEmbedding, word, n=10)
    metrics = wv.embeddings'*get_vector(wv, word)
    topn_positions = sortperm(metrics[:], rev = true)[1:n]
    topn_metrics = metrics[topn_positions]
    return topn_positions, topn_metrics
end

function plot_similarity(wv::WordEmbedding, word, n=10)
    # Get similarities
    positions, metrics = get_similarity(wv, word, n)
    
    # Get the actual words
    similar_words = [wv.words[i] for i in positions]
    
    # Create horizontal bar plot
    p = bar(
        reverse(similar_words),  # Reverse to show most similar at top
        reverse(metrics),
        orientation=:horizontal,
        title="Words Similar to '$(word)'",
        xlabel="Cosine Similarity",
        ylabel="Words",
        legend=false
    )
    return p
end


# # Generate a WordVectors object from text file
# function _from_text(::Type{T}, filename::AbstractString, normalize::Bool=true, delim::Char=' ',fasttext::Bool=false) where T<:Real
#     open(filename) do f
#         header = strip(readline(f))
#         vocab_size,vector_size = map(x -> parse(Int, x), split(header, delim))
#         vocab = Vector{String}(undef, vocab_size)
#         vectors = Matrix{T}(undef, vector_size, vocab_size)
#         for (i, line) in enumerate(readlines(f))
#             !fasttext && (line = strip(line))
#             parts = split(line, delim)
#             word = parts[1]
#             vector = map(x-> parse(T, x), parts[2:end])
#             vocab[i] = word
#             if normalize
#                 vector = vector ./ norm(vector)  # unit vector
#             end
#             vectors[:, i] = vector
#         end
#         return WordVectors(vocab, vectors)
#     end
# end

# # Generate a WordVectors object from binary file
# function _from_binary(::Type{T},filename::AbstractString,skip::Bool=true,normalize::Bool=true,delim::Char=' ') where T<:Real
#     sb = ifelse(skip, 0, 1)
#     open(filename) do f
#         header = strip(readline(f))
#         vocab_size,vector_size = map(x -> parse(Int, x), split(header, delim))
#         vocab = Vector{String}(undef, vocab_size)
#         vectors = zeros(T, vector_size, vocab_size)
#         binary_length = sizeof(Float32) * vector_size
#         for i in 1:vocab_size
#             vocab[i] = strip(readuntil(f, delim))
#             vector = reinterpret(Float32, read(f, binary_length))
#             if normalize
#                 vector = vector ./ norm(vector)  # unit vector
#             end
#             vectors[:,i] = T.(vector)
#             read(f, sb) # new line
#         end
#         return WordVectors(vocab, vectors)
#     end
# end


# function load_text_model(filename::String)
#     # Read lines from file into a vector of strings.
#     println("Reading file...")
#     lines = readlines(filename)

#     println("Parsing header...")
#     # Parse first line to get vocab and vec size. First line is expected to have two integers separated by a space.
#     vocab_size, vector_size = parse.(Int, split(lines[1]))

#     println("Initializing structures...")
#     # Dictionary to store vocabulary. Dict maps vocab to its index in the vectors matrix 
#     vocab = Dict{String, Int}()

#     # Initialize a matrix to store the word vectors with 'vector_size' rows (one for each dimension of the vector) and 'vocab_size' columns (one for each word in the vocabulary).
#     vectors = Matrix{Float64}(undef, vector_size, vocab_size)

#     println("Processing vectors...")
#     # Iterate over each line starting from the second line (i.e., the word vectors).
#     for (i, line) in enumerate(@view lines[2:end])
#         # A valid line should have (vector_size + 1) elements: one word and its vector. Skips line if malformed (doesn't have the correct number of elements)
#         if length(split(line)) != vector_size + 1
#             continue
#         end

#         # Split the line into parts since the first part is the word, and the rest are the vector components.
#         parts = split(line)
#         word = parts[1]                       # The word is the first element.
#         vector = parse.(Float64, parts[2:end]) # Parse the vector components as Float64.

#         # Proceed if the parsed vector has the expected size
#         if length(vector) == vector_size
#             vocab[word] = i                  # Map the word to its index in the dictionary.
#             vectors[:, i] = vector           # Store the vector in the matrix at the corresponding column.
#         end
#     end

#     println("Completed processing vectors from .vec file")
#     # Return a Word2VecModel object containing the vocabulary dictionary and the vectors matrix.
#     return Word2VecModel(vocab, vectors)
# end

# """
# The function retrieves the embedding vector for a specific word based on a pre-loaded Word2VecModel
# """

# function get_word_embedding(model::Word2VecModel, word::String)
#     # Check if the word exists in the model's vocabulary and raise an error if not found
#     if !haskey(model.vocab, word)
#         throw(KeyError("Word '$word' not found in vocabulary"))
#     end

#     # Function looks up word's index in the vocab dict and accesses the corresponding column in the vectors matrix. Then returns word embedding vector for the word
#     return model.vectors[:, model.vocab[word]]
# end

# """
# Still figuring out this shit
# """

# function load_fasttext_embeddings(file_name::String)
#     open(file_name, "r") do f
#         println("Reading file...")
#         # Original working header skipping
#         skip(f, 8)
#         # The ltoh function converts a little-endian representation into the byte order used by your host system.
#         # Little-Endian: The least significant byte (LSB) is stored first in memory. This is common on x86 architectures.
#         # Big-Endian: The most significant byte (MSB) is stored first.
#         dim = ltoh(read(f, Int32))
#         skip(f, 96)  # Changed from 100 to get to word start
        
#         # Constants
#         vocab_size = 2_519_370
        
#         # Initialize storage
#         vocab = Dict{String, Int}()
#         vectors = Matrix{Float32}(undef, dim, vocab_size)
        
#         # Special handling for first two tokens
#         vocab[","] = 1
#         vocab["."] = 2
#         skip(f, 20)  # Skip their metadata
        
#         # Read remaining words (starting from 3)
#         for i in 3:vocab_size
#             # Original working word reading logic
#             while !eof(f) && read(f, UInt8) == 0x00
#                 continue
#             end
#             skip(f, -1)
            
#             word_bytes = UInt8[]
#             while !eof(f)
#                 byte = read(f, UInt8)
#                 if byte == 0x00
#                     break
#                 end
#                 push!(word_bytes, byte)
#             end
            
#             word = String(word_bytes)
#             vocab[word] = i
#             skip(f, 9)
            
#         end
        
#         # Original working vector reading logic
#         pos = position(f)
#         alignment = (4 - pos % 4) % 4
#         if alignment > 0
#             skip(f, alignment)
#         end
        
#         for i in 1:vocab_size
#             for j in 1:dim
#                 vectors[j, i] = ltoh(read(f, Float32))
#             end
#         end
    
#         println("\nCompleted loading $(length(vocab)) words with dimension $dim from .bin file")
        
#         return Word2VecModel(vocab, Float64.(vectors))
#     end
# end
