using Statistics, Plots, LinearAlgebra

"""
This function reduces the dimension of a matrix from NxM to Nx"number_of_pc" with a PCA. 
It returns the projected data.
"""
function reduce_to_2d(data::Matrix, number_of_pc::Int=2)::Matrix
        # Center the data
        num = size(data)[1]
        c_data = data .- mean(data, dims=1)

        # c_data = data .- sum(data, dims = 1) ./ num

        # Compute SVD for stability instead of eigen decomposition        
        U, S, Vt = svd(c_data)

        # Compute the covariance matrix
        #cov_matrix = cov(c_data)

        # Perform eigen decomposition
        #eigen_vals, eigen_vecs = eigen(cov_matrix)

        # Sort eigenvalues (and corresponding eigenvectors) in descending order
        #idx = sortperm(eigen_vals, rev=true)
        #eigen_vecs = eigen_vecs[:, idx]

        # Select the top 2 principal components
        pca_components = Vt[1:number_of_pc, :]
        #pca_components = eigen_vecs[:, 1:number_of_pc]

        # Project the data onto the top 2 principal components
        projected_data = (pca_components * c_data')  # Ensures output is 2×N
        # projected_data = pca_components' * c_data'  
        return projected_data
end


"""
This Function creates a plot of the relations of the arguments like this:
arg1==>arg2,
arg3==>arg4,
...
Note: Use an even number of inputs!
"""
function show_relations(words::String...; wv::WordEmbedding, save_path::String="word_relations.png")
    # Check input - word_count should only be used inside the function
    word_count = length(words)
    if Bool(word_count%2)
        throw(error("need words in multiples of 2 but $word_count are given"))
    end
    
    # Check if all words exist in the embedding
    for word in words
        if !haskey(wv.word_indices, word)
            throw(error("Word '$word' not found in embeddings"))
        end
    end
    # Get embeddings by looking up each word's index and getting its vector
    embeddings = hcat([wv.embeddings[:, wv.word_indices[word]] for word in words]...)                
    # embeddings = reduce(vcat, transpose.([wv.embeddings[:, wv.word_indices[word]] for word in words]))

    labels = text.(words, :bottom)

    #labels = text.([word for word in words], :bottom)    
    
    # reduce dimension
    projection = reduce_to_2d(embeddings)
    
    # preparation for plotting the arrows, infill with zeros and split x, y
    #arrows = [projection[:, 2*i]-projection[:, 2*i-1] for i in 1:Int(word_count/2)]
    arrows = projection[:, 2:2:end] .- projection[:, 1:2:end]
    arrows_x, arrows_y = eachrow(arrows)    
        
    arrows_x = vcat(arrows[1, :], zeros(size(arrows, 2)))
    arrows_y = vcat(arrows[2, :], zeros(size(arrows, 2)))

    #arrows_x = [Bool(i%2) ? arrows[Int(i/2+0.5)][1] : 0 for i in 1:length(arrows)*2]
    #arrows_y = [Bool(i%2) ? arrows[Int(i/2+0.5)][2] : 0 for i in 1:length(arrows)*2]
        
    p = scatter(projection[1, :], projection[2, :], 
         title="PCA Projection to 2D",
         xlabel="First Principal Component",
         ylabel="Second Principal Component",
         legend=false, series_annotations=labels)


    # plot the arrows
    quiver!(p, projection[1, :], projection[2, :], quiver=(arrows_x, arrows_y))
    
    # Save the plot
    savefig(p, save_path)
    
    return p  # Optionally return the plot object
end
