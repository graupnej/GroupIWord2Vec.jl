# Getting Started

## 1) How to install, build and load the package
To install the package, use Julia's package manager:

       using Pkg
       Pkg.add(GroupIWord2Vec)
       
Move to the GroupIWord2Vec directory and activate a project environment in this directory:

       Pkg.activate(".")  # Activate the local environment
       Pkg.instantiate()  # Install dependencies from the Manifest.toml

This instructs Julia to use the Project.toml and Manifest.toml files in the current directory for managing dependencies.
In Julia's package management system (Pkg mode, entered by typing ]) trigger the build process for the package:

       build GroupIWord2Vec

Within the Julia REPL, load the package into the current session

       using GroupIWord2Vec

## 2) Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles # Provides functionality for reading and writing delimited text files
       LinearAlgebra  # Offers a suite of mathematical tools and operations for linear algebra

## 3) Examples
Select a pre-trained Word2Vec model from a reputable source e.g. FastText (binary & text format file):

       https://fasttext.cc/docs/en/pretrained-vectors.html

Load a pre-trained model from a file in text format:

       model = load_text_model("path/to/model.vec")

Alternatively, load a pre-trained model from a file in binary format:

       ...

Generate a word embedding for a given word using the loaded model:

       embedding = get_word_embedding(model, "example")

## 4) References
