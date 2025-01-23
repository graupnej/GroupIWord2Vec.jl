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
