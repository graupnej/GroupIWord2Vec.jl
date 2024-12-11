function load_text_model(filename::String)
    lines = readlines(filename)
    vocab_size, vector_size = parse.(Int, split(lines[1]))
    
    vocab = Dict{String, Int}()
    vectors = Matrix{Float64}(undef, vector_size, vocab_size)
    
    for (i, line) in enumerate(@view lines[2:end])
        # Skip if line is malformed
        if length(split(line)) != vector_size + 1
            continue
        end
        
        parts = split(line)
        word = parts[1]
        vector = parse.(Float64, parts[2:end])
        if length(vector) == vector_size
            vocab[word] = i
            vectors[:, i] = vector
        end
    end
    
    return Word2VecModel(vocab, vectors)
end

function load_binary_model(filename::String)
    open(filename, "r") do f
        # Skip magic number
        skip(f, 4)
        
        # Read dimensions
        header_dim = read(f, Int32)  # 11 (header info)
        vector_dim = read(f, Int32)  # 300 (vector dimension)
        
        # Skip the known pattern of 5s and small numbers (7 Int32s)
        skip(f, 7 * sizeof(Int32))
        
        # Read what we think is vocabulary size
        vocab_size = read(f, Int32)
        println("Reading model with:")
        println("Vector dimension: ", vector_dim)
        println("Vocabulary size: ", vocab_size)
        
        # Initialize storage
        vocab = Dict{String, Int}()
        vectors = zeros(Float64, vector_dim, vocab_size)
        
        # Try reading first few entries
        for i in 1:min(5, vocab_size)
            # Try reading word length
            word_len = read(f, Int32)
            if word_len > 0 && word_len < 100  # sanity check
                # Read word
                word_bytes = read(f, word_len)
                word = String(word_bytes)
                
                # Read vector
                vector = Vector{Float32}(undef, vector_dim)
                read!(f, vector)
                
                println("\nWord $i: ", word)
                println("Vector starts with: ", vector[1:5])
                
                # Store in our data structures
                vocab[word] = i
                vectors[:, i] = Float64.(vector)
            end
        end
        
        return Word2VecModel(vocab, vectors)
    end
end


function get_word_embedding(model::Word2VecModel, word::String)
    if !haskey(model.vocab, word)
        throw(KeyError("Word '$word' not found in vocabulary"))
    end
    return model.vectors[:, model.vocab[word]]
end