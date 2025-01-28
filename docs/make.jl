using Documenter
using GroupIWord2Vec

DocMeta.setdocmeta!(GroupIWord2Vec, :DocTestSetup, :(using GroupIWord2Vec); recursive=true)

makedocs(
    modules = [GroupIWord2Vec],                  
    authors="Julian Graupner <...>, Pablo Ramos Erpenbeck <...>, Knut Bunge <...>, Ladislaus Finger <...e>",
    sitename = "GroupIWord2Vec.jl",  
    format=Documenter.HTML(;
        canonical="https://graupnej.github.io/GroupIWord2Vec.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",                     
    ],
)

deploydocs(;
    repo="graupnej/GroupIWord2Vec.jl",
    devbranch="main",
)
