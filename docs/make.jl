using Documenter
using GroupIWord2Vec  # Replace this with your package name

makedocs(
    modules = [GroupIWord2Vec],                   # Include the modules you want to document
    authors="Julian Graupner <...>, Pablo Ramos Erpenbeck <...>, Knut Bunge <...>, Ladislaus Finger <...e>",
    sitename = "GroupIWord2Vec.jl Documentation",  # Replace with your documentation title
    format = Documenter.HTML(),                   # Generate HTML docs
    pages = [
        "Home" => "index.md",                     # Define the pages of the documentation
    ],
    clean = true,                                 # Clean previously built files
)
