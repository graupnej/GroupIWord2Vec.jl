module GroupIWord2Vec

using LinearAlgebra
using DelimitedFiles

export Word2VecModel, load_binary_model, load_text_model, get_word_embedding
struct Word2VecModel
    vocab::Dict{String, Int}
    vectors::Matrix{Float64}
end

include("functions.jl")

end
