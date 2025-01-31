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
    get_vector_operation(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}, operator::String} -> Vector{Float64}, Float64

Finds the top `n` most similar words to a given word or vector.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `inp1::Union{String, Vector{Float64}}`: First word or vector.
- `inp2::Union{String, Vector{Float64}}`: Second word or vector.
- `operator::String`: The operator string to define the calculation.
operators can be:
"+" -> sum,
"-" -> subtraction,
"*" -> dot product/ cosine similarity,
"euclid" -> Euclidean distance

# Returns
- `Vector{Float64}`: For operations with vecctorial result: '+' and '-'
- `Float64`: For operations with scalar result: '*' and 'euclid'

# Example
```julia
similar_words = get_similar_words(wv, "king", 5)
```
"""
function get_vector_operation(ww::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, operator::String)
    # # Converts both inputs to corresponding vectors using existing function
    inp1_vec = get_any2vec(ww, inp1)
    inp2_vec = get_any2vec(ww, inp2)
    
    # Validate operator
    valid_operators = ["+", "-", "cosine", "euclid"]
    if !(operator in valid_operators)
        throw(ArgumentError("Invalid operator. Please use one of: " * join(valid_operators, ", ")))
    end
    
    # Distinguishes between operations and computes operation
    if operator == "+"
        return inp1_vec + inp2_vec
    elseif operator == "-"
        return inp1_vec - inp2_vec
    elseif operator == "cosine"
        return dot(inp1_vec, inp2_vec) / (norm(inp1_vec) * norm(inp2_vec))
    elseif operator == "euclid"
        return norm(inp1_vec - inp2_vec)
    end
end

"""
    get_similar_words(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}, n::Int=10) -> Vector{String}

Finds the top `n` most similar words to a given word or vector.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `word_or_vec::Union{String, Vector{Float64}}`: A word or an embedding vector.
- `n::Int`: Number of similar words to return (default: 10).

# Returns
- `Vector{String}`: List of most similar words.

# Example
```julia
similar_words = get_similar_words(wv, "king", 5)
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
