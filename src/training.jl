using Flux, ProgressMeter
using Flux: train!
using Random
using OneHotArrays
using Statistics

#load corpus
    #lower case
    #clean up
#->returns text
function load_corpus(path::String)::Vector
    full_text = join(readlines(path), "")
    #remove case sensitivity
    lowercase_text = lowercase(full_text)
    #remove "non-words"
    lowercase_clean_text = replace(lowercase_text, "," => "", ";" => "")    #, "\"" => "", "'" => "")
    
    texts = [String(word) for word in split(lowercase_clean_text, ['.', '!', '?'], keepempty=false)]
    
    #use set datastructure to create vocabulary
    #vocabulary = Set(split(lowercase_clean_text, " "; keepempty=true))

    return texts
end

#tokenize 
#->vocabulary(word::index), text as vector with index eg: [1, 3, 2, 8, 69, 21, 3] (sequences)

function create_vocab_and_index_text(text_lines::Vector)
    vocabulary = Set([String(word) for word in split(join(text_lines, " "), " "; keepempty=false)])
    words = [vocabulary...]
    sequences = [[findfirst(==(String(word)), words) for word in split(line, " ")] for line in text_lines]
    
    return words, sequences
end


#create model(embedding size, vocabulary)
#->embedding Matrixlength(words)

function create_model(embedding_dim, words)
    vocab_size = length(words)
    embeddings = Flux.Embedding(vocab_size => embedding_dim)
    lambda = x -> mean(x , dims=2)
    decode = Dense(embedding_dim => vocab_size, bias=false)
    output = x-> softmax(x)
    
    return  Chain(embeddings, lambda, decode, output)
end


function train_model(epochs::Int, model::Chain, window_size::Int, sequences, opt=Descent())
    word_count = size(model[1].weight)[2]
    loss(model, input, target) = Flux.Losses.crossentropy(target, reshape(model(input), size(target)))

    @showprogress dt=1 desc="Train embedding" for epoch in 1:epochs
        
        context = Array{Int}(undef, window_size*2, 0)
        target = Array{Float32}(undef, word_count, 0)
        for sequence in sequences
            length(sequence)<=2*window_size ? continue : 
            for i in window_size+1:length(sequence)-window_size
                context = hcat(context, vcat(sequence[i-window_size:i-1], sequence[i+1:i+window_size]))
                target = hcat(target, onehot(sequence[i], 1:word_count))
            end

        end
        data=[(context, target)]
        train!(loss, model, data, opt)            

    
    end
    return model    
end

####basic examples
text = load_corpus("data/example.txt")
vocab, sequence = create_vocab_and_index_text(text)
my_model = create_model(10, vocab)
my_model = train_model(1000, my_model, 2, sequence)
trained_embeddning = my_model[1]
