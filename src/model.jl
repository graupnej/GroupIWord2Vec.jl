export WordEmbedding

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
    WordEmbedding

A structure for storing and managing word embeddings, where each word is associated with a vector representation.

# Fields
- `words::Vector{String}`: List of all words in the vocabulary
- `embeddings::Matrix{Float64}`: Matrix where each column is a word's vector representation
- `word_indices::Dict{String, Int}`: Dictionary mapping words to their positions in the vocabulary

# Constructor
    WordEmbedding(words::Vector{String}, matrix::Matrix{Float64})

Creates a WordEmbedding with the given vocabulary and corresponding vectors.

# Throws
- `ArgumentError`: If the number of words doesn't match the number of vectors (matrix columns)

# Example
```julia
# Create a simple word embedding with 2D vectors
words = ["cat", "dog", "house"]
vectors = [0.5 0.1 0.8;
          0.2 0.9 0.3]
embedding = WordEmbedding(words, vectors)
"""
struct WordEmbedding
    # List of all words in the vocabulary. e.g. ["cat", "dog", "house"]
    words::Vector{String}
    
    # Matrix containing all word vectors where each column is one word's vector of numbers
    # Size is (vector_dimension × vocabulary_size). For e.g. 3 words and vectors of length 4 we get a 4×3 matrix
    embeddings::Matrix{Float64}
    
    # Dictionary for fast word lookup. Maps each word to its position in the words vector and embeddings matrix
    word_indices::Dict{String, Int}
    
    # Makes sure the data is valid
    function WordEmbedding(words::Vector{String}, matrix::Matrix{Float64})
        # Validate that number of words matches number of vectors
        if length(words) != size(matrix, 2)
            throw(ArgumentError("Number of words ($(length(words))) must match matrix columns ($(size(matrix, 2)))"))
        end
        
        new(copy(words), copy(matrix), Dict{String, Int}(word => idx for (idx, word) in enumerate(words)))
    end
end

"""
    load_embeddings(path::String; format::Symbol=:text, data_type::Type{Float64}=Float64, normalize_vectors::Bool=true, separator::Char=' ', skip_bytes::Int=1)

Loads word embeddings from a text or binary file.

# Arguments
- `path::String`: Path to the embedding file
- `format::Union{:text, :binary}=:text`: File format (`:text` or `:binary`)
- `data_type::Type{Float64}=Float64`: Type of word vectors
- `normalize_vectors::Bool=true`: Normalize vectors to unit length
- `separator::Char=' '`: Word-vector separator in text files
- `skip_bytes::Int=0`: Bytes to skip after each word-vector pair in binary files

# Throws
- `ArgumentError`: If `format` is not `:text` or `:binary`

# Returns
- `WordEmbedding`: The loaded word embeddings structure

# Example

```julia
embedding = load_embeddings("vectors.txt")  # Load text format
embedding = load_embeddings("vectors.bin", format=:binary, data_type=Float64, skip_bytes=1)  # Load binary format
"""
function load_embeddings(path::String; format::Symbol=:text, data_type::Type{Float64}=Float64, normalize_vectors::Bool=true, separator::Char=' ', skip_bytes::Int=1)
     # For a text file use the read_text_format function
    if format == :text
        return read_text_format(path, data_type, normalize_vectors, separator)
     # For a binary file use the read_binary_format function
    elseif format == :binary
        return read_binary_format(path, data_type, normalize_vectors, separator, skip_bytes)
    else
        throw(ArgumentError("Unsupported format: $format. Supported formats are :text and :binary."))
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
function read_text_format(filepath::AbstractString, ::Type{T},separator::Char) where T<:Real
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
function read_binary_format(filepath::AbstractString,::Type{T},separator::Char,skip_bytes::Int) where T<:Real

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

            # Convert to desired number type and store
            vectors[:, i] = T.(vector)
                    
            # Skip extra bytes (like newlines) after each word-vector pair
            read(file, skip_bytes)
        end

        # Return the WordEmbedding object
        return WordEmbedding(words, vectors)
    end
end
