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
    loss(model, input, target) = Flux.Losses.crossentropy(target, model(input))

    @showprogress dt=1 desc="Train embedding" for epoch in 1:epochs
        
        data = Vector{Tuple{Vector{Int64}, OneHotVector{UInt32}}}()
        for sequence in sequences
            length(sequence)<=2*window_size ? continue : 
            for i in (1+window_size):(length(sequence)-window_size)
                context = vcat(sequence[i-window_size:i-1], sequence[i+1:i+window_size])
                
                target = onehot(sequence[i], 1:word_count)
                data = push!(data, (context, target))
            end

        end
        
    
        train!(loss, model, data, opt)            

    
    end
    return model    
end

####basic examples

#lods the text and returns a cleaned up vec(string) each sentence is a element in vector
#eg: ["the quick brown fox jumps over " ⋯ 40 bytes ⋯ "ly under a tree in the sunshine",
#    "the quick brown rabbit hops ove" ⋯ 50 bytes ⋯ "ly under a tree in the sunshine",
#    "a swift grey rabbit leaps past " ⋯ 40 bytes ⋯ "y beneath the shade of the tree"]
text = load_corpus("data/example.txt")


#creates a vocabulary of all words that are contained in the cleaned up text and 
#transforms the text into a sequence of indeces eg:"The quick brown fox jumps over..." -> [25, 16, 30, 28, 24, 2,...]
vocab, sequence = create_vocab_and_index_text(text)

#creates a model using flux 
#Chain(
#   Embedding(35 => 30),                  # 1_050 parameters
#   var"#19#21"(),
#   Dense(30 => 35; bias=false),          # 1_050 parameters
#   var"#20#22"(),
# ) 
# used like this my_model([3, 5, 6]) -> output softmax distribution over all words (wordcount x 1 Matrix)  
my_model = create_model(30, vocab)

#trains the model with with teh sequences from "create_vocab_and_index_text" returns trained model
my_model = train_model(3, my_model, 1, sequence)


#Proof traing works:
quick_index = findfirst(x-> x=="quick", vocab)  #16
the_index = findfirst(x-> x=="the", vocab)      #25
brown_index = findfirst(x-> x=="brown", vocab)  #30

embedding_dim = 10
model = create_model(embedding_dim, vocab)


output_the_brown = model([the_index, brown_index])
println("output before training for quick:   ", output_the_brown[quick_index])

epochs = 10
window_size = 1
model = my_model = train_model(epochs, my_model, window_size, sequence)


output_the_brown = model([the_index, brown_index])
println("output After training for quick:   ", output_the_brown[quick_index])


#after training we get a 99.99% chance that the word between the and brown is quick The quick brown fox