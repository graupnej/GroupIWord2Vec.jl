using Flux, ProgressMeter
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
    
    loss(model, input, target) = Flux.Losses.crossentropy(target, model(input))

    @showprogress dt=1 desc="Train embedding" for epoch in 1:epochs
        
        context = []
        target = []
        for sequence in sequences
            length(sequence)<=2*window_size ? continue : 
            for i in window_size+1:length(sequence)-window_size
                
                push!(context, vcat(sequence[i-window_size:i-1], sequence[i+1:i+window_size]))
                push!(target, sequence[i])
            end

        end
        data=[(context, target)]
        println(data)
        Flux.Train.train!(loss, model, data, opt)            

    
    end
    return model    
end
a = []
push!(a, [1,4])


text = load_corpus("data/example.txt")
vocab, sequence = create_vocab_and_index_text(text)
my_model = create_model(2, 1:5)
typeof(my_model)
my_model([5, 3, 1])
train_model(10, my_model, 1, sequence)