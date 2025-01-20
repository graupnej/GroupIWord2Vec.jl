using Flux, Statistics, Distributed, ProgressMeter
using Random
#using Cuda

function create_vocabulary(text::String)
    """
    strips "." => "", "," => "", "!" => "", "?" => "", ";" => ""
    from text and returns a vocabulary and the cleaned lowercase text  
    """
    #Join text from file
    #text = join(readlines(filename), " ")

    #delete all dots to make "sunshine" and "sunshine." the same word
    lowercase_text = lowercase(text)
    #remove case sensitivity
    lowercase_clean_text = replace(lowercase_text, "." => "", "," => "", "!" => "", "?" => "", ";" => "")    #, "\"" => "", "'" => "")
    
    #use set datastructure to create vocabulary
    vocabulary = Set(split(lowercase_clean_text, " "; keepempty=true))

    return vocabulary, lowercase_text
end


function make_model(text::String, emb_dim::Int)

    
    vocabulary, l_text = create_vocabulary(text)
    
    input_dimensions = length(vocabulary)
    
    sorted_vocabulary = sort(collect(vocabulary)) 

    embedding_matrix = fill(input_dimensions, emb_dim)

    weights = Embedding(input_dimensions, emb_dim)

    #print("i am called from training!!!")

    return weights, sorted_vocabulary, l_text    
end


function train_embedding(train_file::String, embedding_dim::Int, epochs::Int)

    train_text = join(readlines(train_file), " ")
    weights, sorted_vocabulary, l_text = make_model(train_text, embedding_dim)
    tokenized_text = split(replace(l_text, "." => ""))
    token_num = length(tokenized_text)
    vocab_size = length(sorted_vocabulary)

    @showprogress dt=1 desc="Training vocabulary vectors..." for epoch in range(1, epochs)
        
        #create random order
        order = shuffle(1:vocab_size)

        for i in 1:vocab_size
        
            #select word in dataset
            word = sorted_vocabulary[order[i]]
            
            #println("find word: \"$(word)\"")
            
            #select random occurence
            occur = findall(x -> x == word, tokenized_text)
            token_id = rand(occur)

            
            
            #create input with cbow => word : target : word
            if token_id==1 || token_id==token_num
                #catch edge cases
                continue
            end


            predecessor = tokenized_text[token_id-1]
            successor = tokenized_text[token_id+1]
            
            pre_emb = findfirst(x => x == predecessor, sorted_vocabulary)
            suc_emb = findfirst(x => x == successor, sorted_vocabulary)
            
            #println("$(word): The predecessor is $(predecessor) and the successor is $(successor)")
           
            
            
            #compute loss
            
            #update weights
            

        end
    end
    return weights, sorted_vocabulary
end