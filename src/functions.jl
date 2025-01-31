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
    get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, AbstractVector{<:Real}}, n::Int=10) -> Vector{String}

Finds the top `n` most similar words to a given word or vector based on cosine similarity.

# Arguments
- `wv::WordEmbedding`: The word embedding model containing the vocabulary and embeddings.
- `word_or_vec::Union{AbstractString, AbstractVector{<:Real}}`: The target word or embedding vector.
- `n::Int=10`: The number of most similar words to retrieve (default is 10).

# Throws
- `ArgumentError`: If the input word is not found in the vocabulary and is not a valid vector.
- `DimensionMismatch`: If the input vector does not match the embedding dimension.
- `ArgumentError`: If the input vector has zero norm, making similarity computation invalid.

# Returns
- `Vector{String}`: A list of the `n` most similar words ordered by similarity score.

# Example
```julia
similar_words = get_similar_words(model, "cat", 5)
# Example output: ["dog", "kitten", "feline", "puppy", "pet"]

vec = get_word2vec(model, "ocean")
similar_words = get_similar_words(model, vec, 3)
# Example output: ["sea", "water", "wave"]
"""
function get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, AbstractVector{<:Real}}, n::Int=10)
    vec = get_any2vec(wv, word_or_vec)
    vec === nothing && throw(ArgumentError("Word not found in vocabulary, and input is not a valid vector."))

    embedding_dim = size(wv.embeddings, 1)
    length(vec) != embedding_dim && throw(DimensionMismatch("Vector length ($(length(vec))) ≠ embedding dimension ($embedding_dim)"))

    vec_norm = norm(vec)
    vec_norm == 0 && throw(ArgumentError("Input vector has zero norm, cannot compute cosine similarity."))
    vec /= vec_norm  # Normalize input

    embedding_norms = sqrt.(sum(wv.embeddings .^ 2, dims=1))
    embeddings_normed = wv.embeddings ./ max.(embedding_norms, eps())  # Avoid division by zero

    similarities = embeddings_normed' * vec

    # Get top `n+1` indices and exclude the query word if it's in the vocabulary
    top_indices = partialsortperm(similarities, 1:min(n + 1, length(wv.words)), rev=true)
    similar_words = wv.words[top_indices]

    # Exclude the query word itself (if it was in the vocabulary)
    if word_or_vec isa AbstractString
        similar_words = filter(w -> w != word_or_vec, similar_words)
    end

    return similar_words[1:min(n, length(similar_words))]
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
"""
function get_word_analogy(wv::WordEmbedding, inp1::T, inp2::T, inp3::T, n::Int=5) where {T<:Union{AbstractString, AbstractVector{<:Real}}}
    # Ensure n is valid
    if n ≤ 0
        throw(ArgumentError("n must be greater than 0"))
    end

    # Convert inputs to vectors
    vec1, vec2, vec3 = get_any2vec(wv, inp1), get_any2vec(wv, inp2), get_any2vec(wv, inp3)

    # Compute analogy vector (normalized)
    analogy_vec = vec1 - vec2 + vec3
    analogy_vec /= norm(analogy_vec)  # Normalize

    # Compute cosine similarity scores
    similarities = wv.embeddings' * analogy_vec

    # Identify words that should be excluded
    exclude_words = Set(inp for inp in (inp1, inp2, inp3) if inp isa AbstractString && haskey(wv.word_indices, inp))

    # Get top n+exclusion_count most similar words
    top_indices = partialsortperm(similarities, 1:n+length(exclude_words), rev=true)

    # Filter out excluded words
    filtered_indices = filter(idx -> wv.words[idx] ∉ exclude_words, top_indices)[1:min(n, end)]

    # Return top n words
    return wv.words[filtered_indices]
end

