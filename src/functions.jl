"""
The load_text_model function takes a file in text format containing the pre-trained Word2Vec model as input 
and gives an object of type Word2VecModel as output. This object includes a dictionary mapping words
to their indices and a matrix of vectors where each column corresponds to a word's vector
"""

function load_text_model(filename::String)
    # Read lines from file into a vector of strings.
    println("Reading file...")
    lines = readlines(filename)

    println("Parsing header...")
    # Parse first line to get vocab and vec size. First line is expected to have two integers separated by a space.
    vocab_size, vector_size = parse.(Int, split(lines[1]))

    println("Initializing structures...")
    # Dictionary to store vocabulary. Dict maps vocab to its index in the vectors matrix 
    vocab = Dict{String, Int}()

    # Initialize a matrix to store the word vectors with 'vector_size' rows (one for each dimension of the vector) and 'vocab_size' columns (one for each word in the vocabulary).
    vectors = Matrix{Float64}(undef, vector_size, vocab_size)

    println("Processing vectors...")
    # Iterate over each line starting from the second line (i.e., the word vectors).
    for (i, line) in enumerate(@view lines[2:end])
        # A valid line should have (vector_size + 1) elements: one word and its vector. Skips line if malformed (doesn't have the correct number of elements)
        if length(split(line)) != vector_size + 1
            continue
        end

        # Split the line into parts since the first part is the word, and the rest are the vector components.
        parts = split(line)
        word = parts[1]                       # The word is the first element.
        vector = parse.(Float64, parts[2:end]) # Parse the vector components as Float64.

        # Proceed if the parsed vector has the expected size
        if length(vector) == vector_size
            vocab[word] = i                  # Map the word to its index in the dictionary.
            vectors[:, i] = vector           # Store the vector in the matrix at the corresponding column.
        end
    end

    println("Completed processing vectors from .vec file")
    # Return a Word2VecModel object containing the vocabulary dictionary and the vectors matrix.
    return Word2VecModel(vocab, vectors)
end

"""
The function retrieves the embedding vector for a specific word based on a pre-loaded Word2VecModel
"""

function get_word_embedding(model::Word2VecModel, word::String)
    # Check if the word exists in the model's vocabulary and raise an error if not found
    if !haskey(model.vocab, word)
        throw(KeyError("Word '$word' not found in vocabulary"))
    end

    # Function looks up word's index in the vocab dict and accesses the corresponding column in the vectors matrix. Then returns word embedding vector for the word
    return model.vectors[:, model.vocab[word]]
end

"""
Still figuring out this shit
"""

function load_fasttext_embeddings(file_name::String)
    open(file_name, "r") do f
        println("Reading file...")
        # Original working header skipping
        skip(f, 8)
        # The ltoh function converts a little-endian representation into the byte order used by your host system.
        # Little-Endian: The least significant byte (LSB) is stored first in memory. This is common on x86 architectures.
        # Big-Endian: The most significant byte (MSB) is stored first.
        dim = ltoh(read(f, Int32))
        skip(f, 96)  # Changed from 100 to get to word start
        
        # Constants
        vocab_size = 2_519_370
        
        # Initialize storage
        vocab = Dict{String, Int}()
        vectors = Matrix{Float32}(undef, dim, vocab_size)
        
        # Special handling for first two tokens
        vocab[","] = 1
        vocab["."] = 2
        skip(f, 20)  # Skip their metadata
        
        # Read remaining words (starting from 3)
        for i in 3:vocab_size
            # Original working word reading logic
            while !eof(f) && read(f, UInt8) == 0x00
                continue
            end
            skip(f, -1)
            
            word_bytes = UInt8[]
            while !eof(f)
                byte = read(f, UInt8)
                if byte == 0x00
                    break
                end
                push!(word_bytes, byte)
            end
            
            word = String(word_bytes)
            vocab[word] = i
            skip(f, 9)
            
        end
        
        # Original working vector reading logic
        pos = position(f)
        alignment = (4 - pos % 4) % 4
        if alignment > 0
            skip(f, alignment)
        end
        
        for i in 1:vocab_size
            for j in 1:dim
                vectors[j, i] = ltoh(read(f, Float32))
            end
        end
    
        println("\nCompleted loading $(length(vocab)) words with dimension $dim from .bin file")
        
        return Word2VecModel(vocab, Float64.(vectors))
    end
end
