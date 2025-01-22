"""
# Purpose: Get the vector representation of a specific word from the WordEmbedding
"""
function get_vector(wv::WordEmbedding, word)
    # Performance steps are separated using semicolon
    # Step 1: idx = wv.word_indices[word]     --> Checks the word's index in the dictionary
    # Step 2: wv.embeddings[:,idx]            --> Checks column from the embeddings matrix
    idx = wv.word_indices[word]; wv.embeddings[:,idx]
end

"""
# Purpose: Return the cosine similarity value between two words
"""
function cosine_similarity(wv::WordEmbedding, word_1, word_2)
   # 1. Get vector representations for both words 
   # 2. Transpose first vector (') and multiply (*) with second vector
   # 3. Since word vectors are normalized, this dot product directly gives the cosine similarity
   return get_vector(wv, word_1)'*get_vector(wv, word_2)
end

"""
# Purpose: Find the n (default n = 10) most similar words to a given word
"""
function get_similarity(wv::WordEmbedding, word::String, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*get_vector(wv, word)

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

"""
# Purpose: Find the n (default n = 10) most similar words to a given word and return the matching strings
"""
function get_similarity_as_string(wv::WordEmbedding, vec::Vector, n=10::Int)
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
# Purpose: Find the n (default n = 10) most similar words to a given word and return the matching strings
"""
function get_similarity_as_string(wv::WordEmbedding, word::String, n=10::Int)
    # Step 1: Calculate similarity scores for all words
    # - get_vector(wv, word) gets our target word's vector
    # - wv.embeddings' is the transpose of all vectors
    # - Multiplying these gives cosine similarities (because vectors are normalized)
    metrics = wv.embeddings'*get_vector(wv, word)

    # Step 2: Find positions of top n most similar words
    # - sortperm gets the positions that would sort the array
    # - rev = true means sort in descending order (highest similarity first)
    # - [1:n] takes the first n positions
    topn_positions = sortperm(metrics[:], rev = true)[1:n]


    word_str = [wv.words[i] for i in topn_positions]
    # Return both positions and their similarity scores
    return word_str

end
