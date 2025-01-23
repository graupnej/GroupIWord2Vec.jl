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
    # Validate inputs
    if n < 1
        throw(ArgumentError("Number of results (n) must be positive"))
    end
    
    # Verify all words exist in vocabulary
    all_words = [pos_words; neg_words]
    for word in all_words
        if !haskey(wv.word_indices, word)
            throw(ArgumentError("Word '$word' not found in vocabulary"))
        end
    end
    
    # Calculate result vector directly
    result_vector = zeros(size(wv.embeddings, 1))
    for word in pos_words
        result_vector += get_vector_from_word(wv, word)
    end
    for word in neg_words
        result_vector -= get_vector_from_word(wv, word)
    end
    
    # Calculate similarities
    similarities = wv.embeddings'*result_vector
    
    # Exclude input words efficiently using Set
    exclude_set = Set(wv.word_indices[word] for word in all_words)
    
    # Get top n results
    filtered_indices = filter(i -> !(i in exclude_set), sortperm(similarities[:], rev=true))[1:min(n, end)]
    
    return wv.words[filtered_indices]
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


