using Flux
using ProgressMeter
using Flux: train!
#using Random
using OneHotArrays
using Statistics
 
"""
    create_vocabulary(path::String)::Dict{String, Int}

Creates a vocabulary from a textfile with all occuring words.

# Arguments
- `path::String`: Path to the textfile as string

# Returns
- `Dict{String, Int}`: A dictionary with the words and coresponding indices

# Example
```julia
my_vocabulary = create_vocabulary("data/mydataset.txt")
```
"""
function create_vocabulary(path::String)::Dict{String, Int}
    text = lowercase(join(readlines(path), ""))
    set = Set(split(text, r"\W+", keepempty=false))
    vocabulary = Dict(word => i for (i, word) in enumerate(set))
    return vocabulary
end

"""
    sequence_text(path::String, vocabulary::Dict{String, Int})::Vector{Int64}

Transforms a text to a vector of indices that match the words in the vocabulary.

# Arguments
- `path::String`: Path to the textfile as string
- `vocabulary::Dict{String, Int}`: Vocabulary as a look up table

# Returns
- `Vector{Int64}`: A vector of Integers that contains the text in index form eg: [1, 5, 23, 99, 69, ...]

# Example
```julia
sequence = sequence_text("data/mydataset.txt", my_vocabulary)
```
"""
function sequence_text(path::String, vocabulary::Dict{String, Int})::Vector{Int64}
    text = lowercase(join(readlines(path), ""))
    words = split(text, r"\W+", keepempty=false)
    sequence = Vector{Int}()
    
    for word in words
        haskey(vocabulary, word) ? sequence = push!(sequence, vocabulary[word]) : error("The word \"$word\" is not in vocabulary. Make sure the vocabulary was created with create_vocabulary(\"path\") or add \"$word\" by hand")
    end

    return sequence
end


"""
    create_custom_model(embedding_dim::Int, vocabulary_lenght::Int)::Chain

Creates a Flux model for CBOW.

# Arguments
- `embedding_dim::Int`: The wanted dimensionality of the embedding. 10-300 is recommended depending on the complexity and resources.
- `vocabulary_lenght::Int`: Number of words in the vocabulary

# Returns
- `Chain`: A Flux chain with softmax output

# Notes
- Chain can be used like this my model([2, 5, 18 12]) -> returns prediction of word with the context [2, 5, 18 12] as Softmax probability.  

# Example
```julia
my_model = create_custom_model(50, length(my_vocabulary)) 
```
"""
function create_custom_model(embedding_dim::Int, vocabulary_lenght::Int)::Chain
    
    embeddings = Flux.Embedding(vocabulary_lenght => embedding_dim)
    lambda = x -> mean(x , dims=2)
    decode = Dense(embedding_dim => vocabulary_lenght, bias=false)
    output = x-> softmax(x)
    
    return  Chain(embeddings, lambda, decode, output)
end


"""
    train_custom_model(model::Chain, dataset::String, vocabulary::Dict ,epochs::Int, window_size::Int; optimizer=Descent(), batchsize=10)::Chain

Trains a model on a given dataset. 

# Arguments
- `model::Chain`: The Flux chain from create_model.
- `dataset::String`: Path to the dataset. 
- `vocabulary::Dict`: The vocabulary from create_vocabulary()
- `epochs::Int`: Number of desired epochs.
- `window_size::Int`: Window size for the context window. The total window is 2*window size because preceding and following words are used as context.  
- `optimizer=Descent()`: Optimizer from Flux used for training 
- `batchsize=10`: Number of words trained per epoch. If batchsize = 0 all words in dataset get used once per epoch.


# Returns
- `Chain`: The updated Flux Chain after training.

# Notes
- Number of words Trained = epchs*batchsize.

# Example
```julia
my_updated_model = train_custom_model(my_model, "data/my_dataset.txt", my_vocabulary, 10, 1)
```
"""
function train_custom_model(model::Chain, dataset::String, vocabulary::Dict{String, Int}, epochs::Int, window_size::Int; optimizer=Descent(), batchsize=10)::Chain
    #prepare data
    data = Vector{Tuple{Vector{Int64}, OneHotVector{UInt32}}}()
    
    print("\nData gets sequenced. ")
    sequence = sequence_text(dataset, vocabulary)
    print("Finished!\n")
    
    length(sequence)<=2*window_size ? error("window_size of $window_size is to big for dataset. Keep in mind that the window size is covering 2*window_size+1, because it is used before AND after the word.") : 
    vocab_length = length(vocabulary)
    @showprogress 1 "Prepare inputs and targets from sequence" for i in (1+window_size):(length(sequence)-window_size)
        context = vcat(sequence[i-window_size:i-1], sequence[i+1:i+window_size])
        target = onehot(sequence[i], 1:vocab_length)
       
        data = push!(data, (context, target))
    end

    loss(model, input, target) = Flux.Losses.crossentropy(target, model(input))
    
    batches = collect(Flux.DataLoader(data; batchsize, parallel=Bool(Threads.nthreads()-1), partial = true, shuffle=true))
    batch_number = length(batches)
    if batchsize!==0
        @showprogress 1 "Training embeddings" for i in 1:epochs
            if ((i+batch_number)%batch_number) == 0
                batches = collect(Flux.DataLoader(data; batchsize, parallel=Bool(Threads.nthreads()-1), partial = true, shuffle=true))
            end
            batch = batches[(i+batch_number)%batch_number+1]
            train!(loss, model, batch, optimizer)
        end  
    else
        @showprogress 1 "Training embeddings with full dataset" for i in 1:epochs
            
            train!(loss, model, data, optimizer)
        end
    end
    return model
end

"""
    save_custom_model(model::Chain, vocabulary::Dict{String, Int}, path::String)

Saves the model as a txt in the format for `load_embeddings()`.

# Arguments
- `model::Chain`: The Flux chain from create_model.
- `vocabulary::Dict`: The vocabulary from create_vocabulary()
- `path::String`: Path to the file for saving. 

# Notes
- Make sure to choose a file with a .txt ending if you plan to use it with `load_embeddings()`. 
 
# Example
```julia
save_custom_model(my_model, my_vocabulary, "data/saved_embedd.txt")
```
"""
function save_custom_model(model::Chain, vocabulary::Dict{String, Int}, path::String) 
    open(path, "w") do f
        write(f, "$(length(vocabulary)) $(size(model[1].weight)[1])\n")
    end

    f = open(path, "a")
    words = [k for (k, v) in sort(collect(vocabulary), by=x -> x[2])]   
    for (i, word) in enumerate(words)
        write(f, "$word $(join(model[1].weight[:,i], " "))\n")
    end
    close(f)

end
