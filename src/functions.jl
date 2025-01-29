"""
    get_word2vec(wv::WordEmbedding, word::String) -> Vector{Float64}

Retrieves the embedding vector corresponding to a given word.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `word::String`: The word to look up.

# Returns
- `Vector{Float64}`: The embedding vector corresponding to the word.

# Example
```julia
vec = get_word2vec(wv, "apple")
```
"""
function get_word2vec(wv::WordEmbedding, word::String)
    # Retrieve word index but return nothing if word is not found for ArgumentError
    idx = get(wv.word_indices, word, nothing)
    if idx === nothing
        throw(ArgumentError("Word not found in the embeddings vocabulary"))
    end
    # Return vector for given word at index location
    return wv.embeddings[:, idx]
end

"""
    get_vec2word(wv::WordEmbedding, vec::Vector{Float64}) -> String

Retrieves the closest word in the embedding space to a given vector.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `vec::Vector{Float64}`: The embedding vector.

# Returns
- `String`: The word closest to the given vector.

# Example
```julia
word = get_vec2word(wv, some_vector)
```
"""
function get_vec2word(wv::WordEmbedding, vec::Vector{Float64})
    # Computes cosine similarity score between all embedding vectors and input vector
    similarities = wv.embeddings' * vec
    # Finds embedding vecotr with highest score, saves its index and returns word for corresponding index
    idx = argmax(similarities)
    return wv.words[idx]
end

"""
    get_any2vec(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}) -> Vector{Float64}

Converts a word into its corresponding vector or returns the vector unchanged if already provided.
This allows other functions to take both words and vectors as input.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `word_or_vec::Union{String, Vector{Float64}}`: A word or an embedding vector.

# Returns
- `Vector{Float64}`: The corresponding embedding vector.

# Example 1
```julia
banana_vec = get_any2vec(wv, "banana")
```

#Example 2
```julia
banana_vec = get_any2vec(wv, banana_vec)
```
"""
function get_any2vec(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}})
    # Check the type of the input
    if word_or_vec isa String
        return get_word2vec(wv, word_or_vec)
    elseif word_or_vec isa Vector{Float64}
        return word_or_vec
    else
        throw(ArgumentError("Input must be a String (word) or a Vector (embedding)"))
    end
end

"""
    get_cosine_similarity(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}) -> Float64

Computes the cosine similarity between two words or vectors.

# Arguments
- `wv::WordEmbedding`: The word embedding model.
- `inp1::Union{String, Vector{Float64}}`: First word or vector.
- `inp2::Union{String, Vector{Float64}}`: Second word or vector.

# Returns
- `Float64`: Cosine similarity score.

# Example 1
```julia
similarity = get_cosine_similarity(wv, "cat", "dog")
```
# Example 2
```julia
similarity = get_cosine_similarity(wv, vector_embedding_cat, vector_embedding_dog)
```

"""
function get_cosine_similarity(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}})
    # Make sure both inputs are vectors or convert them into vectors
    vec1, vec2 = get_any2vec(wv, inp1), get_any2vec(wv, inp2)
    # Computing the dot-product gives the cosine similarity score
    return dot(vec1, vec2)
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
    return [wv.words[i] for i in top_indices] # Can it be wv.words[top_indices] ?
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
'+' -> sum,
'-' -> subtraction,
'*' -> dot product/ cosine similarity,
'euclid' -> Euclidean distance

# Returns
- `Vector{Float64}`: For operations with vecctorial result: '+' and '-'
- `Float64`: For operations with scalar result: '*' and 'euclid'

# Example
```julia
similar_words = get_similar_words(wv, "king", 5)
```
"""
function get_vector_operation(wv::WordEmbedding, inp1::Union{String, Vector{64}}, inp2::Union{String, Vector{64}}, operator::String)
    # Converts both inputs to corresponding vectors
    inp1_vec = get_any2vec(wv, inp1)
    inp2_vec = get_any2vec(wv, inp2)
    # Distinguishes between operations and computes operation
    if operator == '+'
        return inp1_vec + inp2_vec
    elseif operator == '-'
        return inp1_vec - inp2_vec
    elseif operator == '*'
        return dot(inp1_vec, inp2_vec)
    elseif operator == 'euclid'
        return sqrt(sum(inp1_vec - inp2_vec).^2))
    else
        throw(ArgumentError("wrong or missing operator, please use +, -, * or euclid"))
    end
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
    filtered_indices = filter(i -> !(i in exclude_set),        # Remove input words
                            sortperm(similarities[:], rev=true) # Sort by similarity
                            )[1:min(n, end)]                   # Take top n
return wv.words[filtered_indices]
end
