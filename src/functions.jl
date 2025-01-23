"""
    get_vector_from_word(wv, string)

# Purpose: Get the vector representation of an input word from the WordEmbedding
"""
function get_vector_from_word(wv::WordEmbedding, word)
    # Performance steps are separated using semicolon
    # Step 1: idx = wv.word_indices[word]     --> Checks the word's index in the dictionary
    # Step 2: wv.embeddings[:,idx]            --> Checks column from the embeddings matrix
    idx = wv.word_indices[word]; wv.embeddings[:,idx]
end

"""
    get_word_from_vector(wv, vector)

# Purpose: Get the word representation of an input vector from the WordEmbedding
"""

function get_word_from_vector(wv::WordEmbedding, vector)
    # Find which column in embeddings matches our vector
    # Use findfirst to get the first matching column index
    idx = findfirst(i -> wv.embeddings[:, i] == vector, 1:size(wv.embeddings, 2)); wv.words[idx]
end

"""
    cosine_similarity(wv, string_1, string_2)

# Purpose: Return the cosine similarity value between two words
"""
function cosine_similarity(wv::WordEmbedding, word_1, word_2)
   # 1. Get vector representations for both words 
   # 2. Transpose first vector (') and multiply (*) with second vector
   # 3. Since word vectors are normalized, this dot product directly gives the cosine similarity
   return get_vector_from_word(wv, word_1)'*get_vector_from_word(wv, word_2)
end

"""
    get_top_similarity_of_word(wv, string, int)

# Purpose: Find the n (default n = 10) most similar words to a given word and return the matching strings
"""
function get_top_similarity_of_word(wv::WordEmbedding, word::String, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*get_vector_from_word(wv, word)

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]


    word_str = [wv.words[i] for i in topn_positions]
    # Return both positions and their similarity scores
    return word_str

end

"""
    get_top_similarity_of_vector(wv, vector, int)

# Purpose: Find the n (default n = 10) most similar words to a given vector and return the matching strings
"""
function get_top_similarity_of_vector(wv::WordEmbedding, vec::Vector, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*vec

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]


    word_str = [wv.words[i] for i in topn_positions]
    # Return both positions and their similarity scores
    return word_str

end

"""
    word_analogy(wv::WordEmbedding, pos_words::Vector{String}, neg_words::Vector{String}, n::Int=5)

Finds analogies using vector arithmetic on word embeddings. For example: king - man + woman = queen

# Arguments
- `wv`: WordEmbedding structure containing the embeddings
- `pos_words`: Words to add (e.g., ["king", "woman"])
- `neg_words`: Words to subtract (e.g., ["man"])
- `n`: Number of results to return (default: 5)

# Returns
- Vector of the n most similar words
"""
function word_analogy(wv::WordEmbedding, pos_words::Vector{String}, neg_words::Vector{String}, n::Int=5)
    # Get dimensions
    vec_size = size(wv.embeddings, 1)
    
    # Create matrix for all vectors
    n_total = length(pos_words) + length(neg_words)
    all_vectors = Matrix{Float64}(undef, vec_size, n_total)
    
    # Add positive word vectors
    for (i, word) in enumerate(pos_words)
        all_vectors[:,i] = get_vector_from_word(wv, word)
    end
    
    # Add negative word vectors (with minus sign)
    for (i, word) in enumerate(neg_words)
        all_vectors[:,i+length(pos_words)] = -get_vector_from_word(wv, word)
    end
    
    # Calculate mean vector
    result_vector = vec(mean(all_vectors, dims=2))
    
    # Calculate similarities with all words
    similarities = wv.embeddings'*result_vector
    
    # Get top positions
    top_positions = sortperm(similarities[:], rev=true)
    
    # Remove input words from results
    filter!(idx -> !(wv.words[idx] in [pos_words; neg_words]), top_positions)
    
    # Return top n words
    return [wv.words[i] for i in top_positions[1:n]]
end

"""
# Purpose: Find the n (default n = 10) most similar words to a given word
"""
function get_similarity(wv::WordEmbedding, word::String, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*get_vector_from_word(wv, word)

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]

    # Step 3: Get the similarity scores for these positions
    topn_metrics = metrics[topn_positions]

    # Return both positions and their similarity scores
    return topn_positions, topn_metrics
end

"""
# Purpose: Find the n (default n = 10) most similar words to a given word
"""
function get_similarity(wv::WordEmbedding, vec::Vector, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*vec

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]

    # Step 3: Get the similarity scores for these positions
    topn_metrics = metrics[topn_positions]
    # Return both positions and their similarity scores
    return topn_positions, topn_metrics
end


