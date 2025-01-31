export WordEmbedding

"""
    train_model(train::AbstractString, output::AbstractString; 
                size::Int=100, window::Int=5, sample::AbstractFloat=1e-3,
                hs::Int=0, negative::Int=5, threads::Int=12, iter::Int=5, 
                min_count::Int=5, alpha::AbstractFloat=0.025,
                debug::Int=2, binary::Int=0, cbow::Int=1, 
                save_vocab=Nothing(), read_vocab=Nothing(),
                verbose::Bool=false) -> Nothing

Trains a Word2Vec model using the specified parameters.

# CAUTION! 
This Function can only be used with Linux or MacOS operating systems! MacOS is only supported with Intel processors M1, M2 are not supported!

# Arguments
- `train::AbstractString`: Path to the input text file used for training.
- `output::AbstractString`: Path to save the trained word vectors.
- `size::Int`: Dimensionality of the word vectors (default: 100).
- `window::Int`: Maximum skip length between words (default: 5).
- `sample::AbstractFloat`: Threshold for word occurrence downsampling (default: 1e-3).
- `hs::Int`: Use hierarchical softmax (1 = enabled, 0 = disabled, default: 0).
- `negative::Int`: Number of negative samples (0 = disabled, common values: 5-10, default: 5).
- `threads::Int`: Number of threads for training (default: 12).
- `iter::Int`: Number of training iterations (default: 5).
- `min_count::Int`: Minimum occurrences for a word to be included (default: 5).
- `alpha::AbstractFloat`: Initial learning rate (default: 0.025).
- `debug::Int`: Debugging verbosity level (default: 2).
- `binary::Int`: Save the vectors in binary format (1 = enabled, 0 = disabled, default: 0).
- `cbow::Int`: Use continuous bag-of-words model (1 = CBOW, 0 = Skip-gram, default: 1).
- `save_vocab`: Path to save the vocabulary (default: `Nothing()`).
- `read_vocab`: Path to read an existing vocabulary (default: `Nothing()`).
- `verbose::Bool`: Print training progress (default: `false`).

# Throws
- `SystemError`: If the training process encounters an issue with file paths.
- `ArgumentError`: If input parameters are invalid.

# Returns
- `Nothing`: The function trains the model and saves the output to a file.

# Example
```julia
train_model("data.txt", "model.vec"; size=200, window=10, iter=10)
```
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
```
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
    load_embeddings(path::String; format::Symbol=:text, data_type::Type{Float64}=Float64, separator::Char=' ', skip_bytes::Int=1)

Loads word embeddings from a text or binary file.

# Arguments
- `path::String`: Path to the embedding file
- `format::Union{:text, :binary}=:text`: File format (`:text` or `:binary`)
- `data_type::Type{Float64}=Float64`: Type of word vectors
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
```
"""
function load_embeddings(path::String; format::Symbol=:text, data_type::Type{Float64}=Float64,normalize_vectors::Bool=true, separator::Char=' ', skip_bytes::Int=1)
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
    read_text_format(filepath::AbstractString, ::Type{T}, normalize::Bool, 
                     separator::Char) where T<:Real -> WordEmbedding

Reads word embeddings from a text file and converts them into a `WordEmbedding` object.

# Arguments
- `filepath::AbstractString`: Path to the text file containing word embeddings.
- `T<:Real`: Numeric type for storing embedding values (e.g., `Float32`, `Float64`).
- `normalize::Bool`: Whether to normalize vectors to unit length for comparison.
- `separator::Char`: Character used to separate words and vector values in the file.

# Throws
- `SystemError`: If the file cannot be opened or read.
- `ArgumentError`: If the file format is incorrect or missing data.

# Returns
- `WordEmbedding`: A structure containing words and their corresponding embedding vectors.

# Example
```julia
embeddings = read_text_format("vectors.txt", Float32, true, ' ')
```
"""
function read_text_format(filepath::AbstractString, ::Type{T},normalize::Bool, separator::Char) where T<:Real
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

""""
    read_binary_format(filepath::AbstractString, ::Type{T}, normalize::Bool, 
                       separator::Char, skip_bytes::Int) where T<:Real -> WordEmbedding

Reads word embeddings from a binary file and converts them into a `WordEmbedding` object.

# Arguments
- `filepath::AbstractString`: Path to the binary file containing word embeddings.
- `T<:Real`: Numeric type for storing embedding values (e.g., `Float32`, `Float64`).
- `normalize::Bool`: Whether to normalize vectors to unit length for comparison.
- `separator::Char`: Character separating words and vector data in the file.
- `skip_bytes::Int`: Number of bytes to skip after each word-vector pair (e.g., for handling separators).

# Throws
- `SystemError`: If the file cannot be opened or read.
- `ArgumentError`: If the file format is incorrect or data is missing.

# Returns
- `WordEmbedding`: A structure containing words and their corresponding embedding vectors.

# Example
```julia
embeddings = read_binary_format("vectors.bin", Float32, true, ' ', 1)
```
"""
function read_binary_format(filepath::AbstractString,::Type{T},normalize::Bool ,separator::Char,skip_bytes::Int) where T<:Real

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
