using LinearAlgebra

"""
    get_word2vec(wv::WordEmbedding, word::String) -> Vector{Float64}

Retrieves the embedding vector corresponding to a given word.

# Arguments
- `wv::WordEmbedding`: The word embedding structure containing the vocabulary and embeddings
- `word::String`: The word to look up

# Throws
- `ArgumentError`: If the word is not found in the embedding model

# Returns
- `Vector{Float64}`: The embedding vector of the requested word of type Float64

# Example
```julia
vec = get_word2vec(model, "dog")
```
"""
function get_word2vec(wv::WordEmbedding, word::String)
    # Retrieve word index
    idx = get(wv.word_indices, word, nothing)
    if idx === nothing
        throw(ArgumentError("Word not found in the embeddings vocabulary"))
    end
    # Returns vector for given word at index location
    return Vector(wv.embeddings[:, idx])
end

"""
    get_vec2word(wv::WordEmbedding,vec::Vector{Float64}) -> String

Retrieves the closest word in the embedding space to a given vector based on cosine similarity.

# Arguments
- `wv::WordEmbedding`: The word embedding structure containing the vocabulary and embeddings
- `vec::Vector{Float64}`: A vector representation of a word

# Returns
- `String`: The word from the vocabulary closest to the given vector

# Throws
- `DimensionMismatch`: If the input vector's dimension does not match the word vector dimensions

# Example
```julia
words = ["cat", "dog"]
vectors = [0.5 0.1;
          0.2 0.9]
embedding = WordEmbedding(words, vectors)

get_vec2word(embedding, [0.51, 0.19])  # Returns "cat"
```
"""
function get_vec2word(wv::WordEmbedding, vec::Vector{Float64})
    # Ensure input vector has correct dimension
    if length(vec) != size(wv.embeddings, 1)
        throw(DimensionMismatch("Input vector must have the same dimension as word vectors. Expected $(size(wv.embeddings, 1)), got $(length(vec))."))
    end

    # Normalize input vector
    # vec = vec / norm(vec)

    # Compute cosine similarity with all word embeddings
    similarities = wv.embeddings' * vec

    # Get index of the highest similarity
    idx = argmax(similarities)

    return wv.words[idx]
end

"""
    get_any2vec(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}) -> Vector{Float64} 

Converts a word into its corresponding vector representation or returns the vector unchanged if already provided

# Arguments
- `wv::WordEmbedding`: The word embedding structure containing the vocabulary and embeddings
- `word_or_vec::Union{String, Vector{Float64}}`: A word to be converted into a vector, or a numerical vector to be validated

# Returns
- `Vector{Float64}`: The vector representation of the word if input is a `String`, or the validated vector

# Throws
- `DimensionMismatch`: If the input vector does not match the embedding dimension.
- `ArgumentError`: If the input is neither a word nor a valid numeric vector.

# Example
```julia
words = ["cat", "dog"]
vectors = [0.5 0.1;
          0.2 0.9]
wv = WordEmbedding(words, vectors)

get_any2vec(wv, "cat")  # Returns [0.5, 0.2]
get_any2vec(wv, [0.5, 0.2])  # Returns [0.5, 0.2]
```
"""
function get_any2vec(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}})
    if word_or_vec isa String
        # Convert word to vector
        return get_word2vec(wv, word_or_vec)
    elseif word_or_vec isa Vector{Float64}
        # Check dimension match for vector input
        if length(word_or_vec) != size(wv.embeddings, 1)
            throw(DimensionMismatch("Input vector dimension $(length(word_or_vec)) does not match embedding dimension $(size(wv.embeddings, 1))"))
        end
        return word_or_vec
    else
        # Explicitly handle invalid input types
        throw(ArgumentError("Input must be a String (word) or a Vector of real numbers matching the embedding dimension."))
    end
end

"""
    get_vector_operation(ww::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, operator::Symbol) -> Union{Vector{Float64}, Float64}

Performs a mathematical operation between two word embedding vectors

# Arguments
- `ww::WordEmbedding`: The word embedding structure containing the vocabulary and embeddings
- `inp1::Union{String, Vector{Float64}}`: The first input, which can be a word (String) or a precomputed embedding vector
- `inp2::Union{String, Vector{Float64}}`: The second input, which can be a word (String) or a precomputed embedding vector
- `operator::Symbol`: The operation to perform. Must be one of `:+`, `:-`, `:cosine`, or `:euclid`

# Throws
- `ArgumentError`: If the operator is invalid.
- `ArgumentError`: If cosine similarity is attempted on a zero vector
- `DimensionMismatch`: If the input vectors do not have the same length

# Returns
- `Vector{Float64}`: If the operation is `:+` or `:-`, returns the resulting word vector
- `Float64`: If the operation is `:cosine` or `:euclid`, returns a scalar value

# Example
```julia
vec = get_vector_operation(model, "king", "man", :-)
similarity = get_vector_operation(model, "cat", "dog", :cosine)
distance = get_vector_operation(model, "car", "bicycle", :euclid)
```
"""
function get_vector_operation(ww::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, operator::Symbol)
    # Convert inputs to vectors
    inp1_vec = get_any2vec(ww, inp1)
    inp2_vec = get_any2vec(ww, inp2)

    # Validate dimensions
    if length(inp1_vec) != length(inp2_vec)
        throw(DimensionMismatch("Vectors must have the same length, but got $(length(inp1_vec)) and $(length(inp2_vec))"))
    end

    # Define valid operators as a Set for efficiency
    valid_operators = Set([:cosine, :euclid, :+, :-])
    
    if operator ∉ valid_operators
        throw(ArgumentError("Invalid operator. Use one of: " * join(string.(collect(valid_operators)), ", ")))
    end

    # Perform the operation
    return if operator == :+
        inp1_vec + inp2_vec
    elseif operator == :-
        inp1_vec - inp2_vec
    elseif operator == :cosine
        norm1 = norm(inp1_vec)
        norm2 = norm(inp2_vec)
        if norm1 ≈ 0 || norm2 ≈ 0
            throw(ArgumentError("Cannot compute cosine similarity for zero vectors"))
        end
        dot(inp1_vec, inp2_vec) / (norm1 * norm2)
    elseif operator == :euclid
        norm(inp1_vec - inp2_vec)
    end
end

"""
    get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, AbstractVector{<:Real}}, n::Int=10) -> Vector{String}

Finds the `n` most similar words to a given word or vector based on cosine similarity.

# Arguments
- `wv`: The word embedding model.
- `word_or_vec`: The target word or embedding vector.
- `n`: Number of similar words to return (default: 10).

# Throws
- `ArgumentError`: If `n` is not positive, the word is missing, or the vector has zero norm.
- `DimensionMismatch`: If the vector size is incorrect.

# Returns
A list of `n` most similar words, sorted by similarity.

# Example
```julia
get_similar_words(model, "cat", 5)  # ["dog", "kitten", "feline", "puppy", "pet"]
get_similar_words(model, get_word2vec(model, "ocean"), 3)  # ["sea", "water", "wave"]
```
"""
function get_similar_words(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}, n::Int=10)
    # Make sure input is a vector or convert it into a vector
    vec = get_any2vec(wv, word_or_vec)
    # Computes cosine similarity score between all embedding vectors and input vector
    similarities = wv.embeddings' * vec
    # Sort similarities for highest n cosine similarity scores
    top_indices = sortperm(similarities[:], rev=true)[1:n]
    return wv.words[top_indices]
end

"""
    get_word_analogy(wv::WordEmbedding, inp1::T, inp2::T, inp3::T, n::Int=5) where {T<:Union{AbstractString, AbstractVector{<:Real}}} -> Vector{String}

Finds the top `n` words that best complete the analogy: `inp1 - inp2 + inp3 = ?`.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `inp1, inp2, inp3::T`: Words or vectors for analogy computation.
- `n::Int=5`: Number of closest matching words to return.

# Returns
- `Vector{String}`: A list of the top `n` matching words.

# Notes
- Input words are converted to vectors automatically.
- The computed analogy vector is normalized.
- Input words (if given as strings) are excluded from results.

# Example
```julia
get_word_analogy(model, "king", "man", "woman", 3) 
# → ["queen", "princess", "duchess"]
```
"""
function get_word_analogy(
    wv::WordEmbedding, 
    inp1::Union{AbstractString, AbstractVector{<:Real}}, 
    inp2::Union{AbstractString, AbstractVector{<:Real}}, 
    inp3::Union{AbstractString, AbstractVector{<:Real}}, 
    n::Int=5; 
    metric::Symbol=:cosine  # Choose similarity metric: :cosine (default) or :dot_product
)
    # Ensure valid n
    if n ≤ 0
        throw(ArgumentError("n must be greater than 0"))
    end

    # Convert inputs to vectors
    vec1, vec2, vec3 = get_any2vec(wv, inp1), get_any2vec(wv, inp2), get_any2vec(wv, inp3)

    # Compute analogy vector
    analogy_vec = vec1 - vec2 + vec3

    # Apply normalization if using cosine similarity
    if metric == :cosine
        analogy_vec /= norm(analogy_vec)
    elseif metric != :dot_product
        throw(ArgumentError("Invalid metric: choose either :cosine or :dot_product"))
    end

    # Compute similarity scores using view to avoid unnecessary memory allocation
    similarities = view(wv.embeddings, :, :)' * analogy_vec

    # Get words for input vectors
    word1, word2, word3 = get_vec2word(wv, vec1), get_vec2word(wv, vec2), get_vec2word(wv, vec3)

    # Create exclusion set safely
    exclude_set = Set(word for word in (word1, word2, word3) if word !== nothing && haskey(wv.word_indices, word))

    # Get top n+exclusion_count most similar words
    top_indices = partialsortperm(similarities, 1:n+length(exclude_set), rev=true)

    # Filter out excluded words
    filtered_indices = filter(idx -> wv.words[idx] ∉ exclude_set, top_indices)[1:min(n, end)]

    # Return top n words
    return wv.words[filtered_indices]
end


#function get_word_analogy(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, inp3::Union{String, Vector{Float64}}, n::Int=5)
    # Get vectors for all inputs for vector calculations
    #vec1, vec2, vec3 = get_any2vec(wv, inp1), get_any2vec(wv, inp2), get_any2vec(wv, inp3)
    # Get words for all inputs for excluding in result
    #word1, word2, word3 = get_vec2word(wv,vec1), get_vec2word(wv,vec1), get_vec2word(wv,vec1)
    # Make a list of all input words
    #all_words = [word1, word2, word3]
    # Compute analogy vector
    #analogy_vec = vec1 - vec2 + vec3
    # Compute the cosine similarity score for each embedding vector with the analogy vector
    #similarities = wv.embeddings' * analogy_vec
    # Make a set including all input words
    #exclude_set = Set(wv.word_indices[word] for word in all_words)
    # Search for n vectors with highest similarity score excluding input words
    #filtered_indices = first(filter(!in(exclude_set), sortperm(similarities[:], rev=true)))[1:min(n, end)]                   # Take top n
#return wv.words[filtered_indices]
#end
