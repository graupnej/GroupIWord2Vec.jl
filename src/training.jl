using Flux, ProgressMeter
using Flux: train!
using Random
using OneHotArrays
using Statistics
using BenchmarkTools


"""
    create_vocabulary(path::String)::Dict{String, Int}

Creates a vocabulary from a textfile with all occuring words.

# Arguments
- `path::String`: Path to the textfile as string

# Throws
- `ArgumentError`: If the word is not found in the embedding model.

# Returns
- `Dict{String, Int}`: A dictionary with the words and coresponding indices

# Example
```julia
vocabulary = create_vocabulary("data/mydataset.txt")
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

Transforms a text to a vector of indices that match with the words in the vocabulary.

# Arguments
- `path::String`: Path to the textfile as string
- `vocabulary::Dict{String, Int}`: Path to the textfile as string

# Throws
- `ArgumentError`: If the word is not found in the embedding model.

# Returns
- `Dict{String, Int}`: A dictionary with the words and coresponding indices

# Example
```julia
vocabulary = create_vocabulary("data/mydataset.txt")
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

function create_model(embedding_dim::Int, vocabulary_lenght::Int)::Chain
    
    embeddings = Flux.Embedding(vocabulary_lenght => embedding_dim)
    lambda = x -> mean(x , dims=2)
    decode = Dense(embedding_dim => vocabulary_lenght, bias=false)
    output = x-> softmax(x)
    
    return  Chain(embeddings, lambda, decode, output)
end

function train_model(model::Chain, dataset::String, vocabulary::Dict ,epochs::Int, window_size::Int; optimizer=Descent(), batchsize=10)::Chain
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



#test
task = "easy"

task=="easy" ? data = "data/example.txt" : data = "data/text8"
vocab = create_vocabulary(data)
model = create_model(10, length(vocab))
data
new_model = train_model(model, data, vocab, 10, 1, batchsize=10)
typeof(sequence_text("data/example.txt", vocab))

vocab["the"]
vocab["brown"]
vocab["quick"]
new_model([25, 30])[16]

# #Proof traing works:
# quick_index = findfirst(x-> x=="quick", vocab)  #16
# the_index = findfirst(x-> x=="the", vocab)      #25
# brown_index = findfirst(x-> x=="brown", vocab)  #30

# embedding_dim = 10
# model = create_model(embedding_dim, vocab)


# output_the_brown = model([the_index, brown_index])
# println("output before training for quick:   ", output_the_brown[quick_index])

# epochs = 10
# window_size = 1
# model = my_model = train_model(epochs, my_model, window_size, sequence)


# output_the_brown = model([the_index, brown_index])
# println("output After training for quick:   ", output_the_brown[quick_index])


# #after training we get a 99.99% chance that the word between the and brown is quick The quick brown fox
