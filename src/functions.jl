"""
    get_word2vec(wv::WordEmbedding, word::String) -> Vector{T}

Retrieves the embedding vector corresponding to a given word.

# Arguments
- `wv::WordEmbedding`: The word embedding model containing the vocabulary and embeddings.
- `word::String`: The word to look up

# Throws
- `ArgumentError`: If the word is not found in the embedding model.

# Returns
- `Vector{T}`: The embedding vector of the requested word, where `T` is the numerical type of the embeddings.

# Example
```julia
vec = get_word2vec(model, "dog")
"""
function get_word2vec(wv::WordEmbedding, word::String)
    # Retrieve word index but return nothing if word is not found for ArgumentError
    idx = get(wv.word_indices, word, nothing)
    if idx === nothing
        throw(ArgumentError("Word not found in the embeddings vocabulary"))
    end
    # Returns (and ensures) vector for given word at index location
    return Vector(wv.embeddings[:, idx])
end

"""
    get_vec2word(wv::WordEmbedding{S, T}, vec::Vector{T}) where {S<:AbstractString, T<:Real} -> String

Retrieves the closest word in the embedding space to a given vector based on cosine similarity.

# Arguments
- `wv::WordEmbedding{S, T}`: A word embedding structure with words and their corresponding vector representations.
- `vec::Vector{T}`: A vector representation of a word.

# Returns
- `S`: The word from the vocabulary closest to the given vector

# Throws
- `DimensionMismatch`: If the input vector's dimension does not match the word vector dimensions.

# Example
```julia
words = ["cat", "dog"]
vectors = [0.5 0.1;
          0.2 0.9]
embedding = WordEmbedding(words, vectors)

get_vec2word(embedding, [0.51, 0.19])  # Returns "cat"
"""
function get_vec2word(wv::WordEmbedding{S, T}, vec::Vector{T}) where {S<:AbstractString, T<:Real}
    # Ensure input vector has correct dimension
    if length(vec) != size(wv.embeddings, 1)
        throw(DimensionMismatch("Input vector must have the same dimension as word vectors. Expected $(size(wv.embeddings, 1)), got $(length(vec))."))
    end

    # Normalize the input vector
    vec = vec / norm(vec)

    # Compute cosine similarity with all word embeddings
    similarities = wv.embeddings' * vec

    # Get the index of the highest similarity
    idx = argmax(similarities)

    return wv.words[idx]
end

"""
    get_any2vec(wv::WordEmbedding{S, T}, word_or_vec::Union{S, Vector{<:Real}}) -> Vector{T} 
    where {S<:AbstractString, T<:Real}

Converts a word into its corresponding vector representation or returns the vector unchanged if already provided.

# Arguments
- `wv::WordEmbedding{S, T}`: A word embedding structure with words and their corresponding vector representations.
- `word_or_vec::Union{S, Vector{<:Real}}`: A word to be converted into a vector, or a numerical vector to be validated.

# Returns
- `Vector{T}`: The vector representation of the word if input is a `String`, or the validated vector (converted to `T` if necessary).

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
"""
function get_any2vec(wv::WordEmbedding{S, T}, word_or_vec::Union{S, Vector{<:Real}}) where {S<:AbstractString, T<:Real}
    if word_or_vec isa S
        # Convert word to vector
        return get_word2vec(wv, word_or_vec)
    elseif word_or_vec isa Vector{<:Real}
        # Check dimension match for vector input
        if length(word_or_vec) != size(wv.embeddings, 1)
            throw(DimensionMismatch("Input vector dimension $(length(word_or_vec)) does not match embedding dimension $(size(wv.embeddings, 1))"))
        end
        return convert(Vector{T}, word_or_vec)
    else
        # Explicitly handle invalid input types
        throw(ArgumentError("Input must be a String (word) or a Vector of real numbers matching the embedding dimension."))
    end
end

"""
    get_vector_operation(ww::WordEmbedding, inp1::Union{String, AbstractVector{<:Real}}, 
                         inp2::Union{String, AbstractVector{<:Real}}, operator::Symbol) -> Union{Vector{<:Real}, Float64}

Performs a mathematical operation between two word embedding vectors.

# Arguments
- `ww::WordEmbedding`: The word embedding model containing the vocabulary and embeddings.
- `inp1::Union{String, AbstractVector{<:Real}}`: The first input, which can be a word (String) or a precomputed embedding vector.
- `inp2::Union{String, AbstractVector{<:Real}}`: The second input, which can be a word (String) or a precomputed embedding vector.
- `operator::Symbol`: The operation to perform. Must be one of `:+`, `:-`, `:cosine`, or `:euclid`.

# Throws
- `ArgumentError`: If the operator is invalid.
- `ArgumentError`: If cosine similarity is attempted on a zero vector.
- `DimensionMismatch`: If the input vectors do not have the same length.

# Returns
- `Vector{<:Real}`: If the operation is `:+` (addition) or `:-` (subtraction), returns the resulting word vector.
- `Float64`: If the operation is `:cosine` (cosine similarity) or `:euclid` (Euclidean distance), returns a scalar value.

# Example
```julia
vec = get_vector_operation(model, "king", "man", :-)
similarity = get_vector_operation(model, "cat", "dog", :cosine)
distance = get_vector_operation(model, "car", "bicycle", :euclid)
"""
function get_vector_operation(ww::WordEmbedding, inp1::Union{String, AbstractVector{<:Real}}, 
                              inp2::Union{String, AbstractVector{<:Real}}, operator::Symbol)
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
    get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, Vector{<:Real}}, n::Int=10) -> Vector{String}

Finds the top `n` most similar words to a given word or vector based on cosine similarity.

# Arguments
- `wv::WordEmbedding`: The word embedding model containing the vocabulary and embeddings.
- `word_or_vec::Union{AbstractString, Vector{<:Real}}`: The input word (from the vocabulary) or a vector of the same dimension as the embeddings.
- `n::Int=10`: The number of most similar words to return. Defaults to `10`.

# Throws
- `ArgumentError`: If the word is not in the vocabulary and the input is not a valid vector.
- `DimensionMismatch`: If the provided vector does not match the embedding dimension.

# Returns
- `Vector{String}`: A list of the top `n` most similar words from the vocabulary.

# Example
```julia
similar_words = get_similar_words(model, "cat", 5)
similar_vectors = get_similar_words(model, [0.2, -0.5, 0.1, 0.8], 5)
"""
function get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, Vector{<:Real}}, n::Int=10)
    # Ensure input vector is valid
    vec = get_any2vec(wv, word_or_vec)
    if vec === nothing
        throw(ArgumentError("Word not found in vocabulary, and input is not a valid vector."))
    end

    # Ensure input vector has correct dimension
    if length(vec) != size(wv.embeddings, 1)
        throw(DimensionMismatch("Input vector length ($(length(vec))) does not match word embedding dimension ($(size(wv.embeddings, 1)))"))
    end

    # Normalize the input vector for cosine similarity
    vec = vec / norm(vec)

    # Normalize all embeddings for cosine similarity
    embeddings_normed = wv.embeddings ./ norm.(eachcol(wv.embeddings))

    # Compute cosine similarity
    similarities = embeddings_normed' * vec

    # Efficiently get the top `n` most similar words
    n = min(n, length(wv.words))  # Avoid requesting more than available words
    top_indices = partialsortperm(similarities, 1:n, rev=true)

    return wv.words[top_indices]
end

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
    get_word_analogy(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, inp3::Union{String, Vector{Float64}}, n::Int=5) -> Vector{String}

Performs word analogy calculations like: `king - queen + woman = man`.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `inp1::Union{String, Vector{Float64}}`: First input (e.g., "king", vector_embedding_king).
- `inp2::Union{String, Vector{Float64}}`: Second input (e.g., "queen", vector_embedding_queen).
- `inp3::Union{String, Vector{Float64}}`: Third input (e.g., "woman", vector_embedding_woman).
- `n::Int`: Number of similar words to return (default: 5).

# Returns
- `Vector{String}`: List of most similar words to the resulting vector.

# Example
```julia
analogy_result = get_word_analogy(wv, "king", "queen", "woman")
```
"""
function get_word_analogy(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, inp3::Union{String, Vector{Float64}}, n::Int=5)
    # Get vectors for all inputs for vector calculations
    vec1, vec2, vec3 = get_any2vec(wv, inp1), get_any2vec(wv, inp2), get_any2vec(wv, inp3)
    # Get words for all inputs for excluding in result
    word1, word2, word3 = get_vec2word(wv,vec1), get_vec2word(wv,vec1), get_vec2word(wv,vec1)
    # Make a list of all input words
    all_words = [word1, word2, word3]
    # Compute analogy vector
    analogy_vec = vec1 - vec2 + vec3
    # Compute the cosine similarity score for each embedding vector with the analogy vector
    similarities = wv.embeddings' * analogy_vec
    # Make a set including all input words
    exclude_set = Set(wv.word_indices[word] for word in all_words)
    # Search for n vectors with highest similarity score excluding input words
    filtered_indices = first(filter(!in(exclude_set), sortperm(similarities[:], rev=true)))[1:min(n, end)]                   # Take top n
return wv.words[filtered_indices]
end
