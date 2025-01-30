using Statistics, Plots, LinearAlgebra

"""
    reduce_to_2d(data::Matrix, number_of_pc::Int=2) -> Matrix{Float64}

Performs Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset NxM to Nx"number_of_pc" and returns a projected data

# Arguments
- `data::Matrix`: The input data matrix where rows represent samples and columns represent features.
- `number_of_pc::Int=2`: The number of principal components to retain (default: 2).

# Returns
- `Matrix{Float64}`: A matrix of shape `(number_of_pc × N)`, where `N` is the number of samples, containing the projected data in the reduced dimensional space.

# Example
```julia
data = randn(100, 50)  # 100 samples, 50 features
reduced_data = reduce_to_2d(data, 2)
"""
function reduce_to_2d(data::Matrix, number_of_pc::Int=2)::Matrix
        # Center the data
        c_data = data .- mean(data, dims=1)

        # Compute the covariance matrix
        cov_matrix = cov(c_data)

        # Perform eigen decomposition
        eigen_vals, eigen_vecs = eigen(Symmetric(cov_matrix))
        
        # Sort eigenvalues (and corresponding eigenvectors) in descending order and select the top "number_of_pc" principal components
        idx = sortperm(eigen_vals, rev=true)
        pca_components = eigen_vecs[:, idx[1:number_of_pc]]

        # Project the data onto the top principal components
        projected_data = pca_components' * c_data'  
        
        return Matrix{Float64}(projected_data)
end


"""
    show_relations(words::String...; wv::WordEmbedding, save_path::String="word_relations.png") -> Plots.Plot

Generates a 2D PCA projection of the given word embeddings and visualizes their relationships like this:
arg1==>arg2,
arg3==>arg4,
...
Note: Use an even number of inputs!

# Arguments
- `words::String...`: A list of words to visualize. The number of words must be a multiple of 2.
- `wv::WordEmbedding`: The word embedding model containing the word vectors.
- `save_path::String="word_relations.png"`: The file path where the generated plot will be saved. If empty or `nothing`, the plot is not saved.

# Throws
- `ArgumentError`: If the number of words is not a multiple of 2.
- `ArgumentError`: If any of the provided words are not found in the embedding model.

# Returns
- `Plots.Plot`: A scatter plot with arrows representing word relationships.

# Example
```julia
p = show_relations("king", "queen", "man", "woman"; wv=model, save_path="relations.png")

"""
function show_relations(words::String...; wv::WordEmbedding, save_path::String="word_relations.png")
    # Check input - word_count should only be used inside the function
    word_count = length(words)
    if word_count % 2 != 0
        throw(ArgumentError("Need words in multiples of 2, but $word_count were given"))
    end
    
    # Create a dictionary mapping words to their indices (or `nothing` if missing)
    indices = Dict(word => get(wv.word_indices, word, nothing) for word in words)

    # Find missing words
    missing_words = [word for (word, idx) in indices if idx === nothing]
    
    # If there are missing words, throw an error listing all of them
    if !isempty(missing_words)
        throw(ArgumentError("Words not found in embeddings: " * join(missing_words, ", ")))
    end
    
    # Get embeddings by looking up each word's index and getting its vector
    embeddings = permutedims(hcat([wv.embeddings[:, indices[word]] for word in words]...))

    labels = text.([word for word in words], :bottom)    

    # reduce dimension
    projection = reduce_to_2d(embeddings)

    # preparation for plotting the arrows, infill with zeros and split x, y
    arrows = [projection[:, 2*i]-projection[:, 2*i-1] for i in 1:Int(word_count/2)]
    arrows_x = [Bool(i%2) ? arrows[Int(i/2+0.5)][1] : 0 for i in 1:length(arrows)*2]
    arrows_y = [Bool(i%2) ? arrows[Int(i/2+0.5)][2] : 0 for i in 1:length(arrows)*2]
        
    p = scatter(projection[1, :], projection[2, :], 
            title="Word Embedding PCA Projection",
            xlabel="First Principal Component",
            ylabel="Second Principal Component",
            legend=false, series_annotations = labels)

    # plot the arrows
    quiver!(p, projection[1, :], projection[2, :], quiver=(arrows_x, arrows_y))
    
    # Save the plot
    if save_path !== nothing && !isempty(save_path)
        savefig(p, save_path)
    end
    
    return p  # Optionally return the plot object
end
