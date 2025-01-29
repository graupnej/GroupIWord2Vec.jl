using Documenter
using GroupIWord2Vec

DocMeta.setdocmeta!(GroupIWord2Vec, :DocTestSetup, :(using GroupIWord2Vec); recursive=true)

makedocs(
    modules = [GroupIWord2Vec],                  
    authors="Julian Graupner <Julian.Graupner@physik.hu-berlin.de>, Pablo Ramos Erpenbeck <p.ramoserpenbeck@campus.tu-berlin.de>, Knut Bunge <knut.c.bunge@campus.tu-berlin.de>, Ladislaus Finger <l.finger@campus.tu-berlin.de>",
    sitename = "GroupIWord2Vec.jl",  
    format=Documenter.HTML(;
        canonical="https://graupnej.github.io/GroupIWord2Vec.jl",
        edit_link="main",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Functions" => "functions.md",
    ],
    assets = String["../../assets/PCAProjection.png"]
)

deploydocs(;
    repo="github.com/graupnej/GroupIWord2Vec.jl",
    devbranch="main",
)
