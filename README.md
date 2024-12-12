# Getting Started

## How to install the package
To install the package, use Julia's package manager:

       using Pkg
       Pkg.add(GroupIWord2Vec)

## How to build the package
Move to the GroupIWord2Vec directory and activate a project environment in this directory:

       Pkg.activate(".") # Activate the local environment
       Pkg.instantiate()  # Install dependencies from the Manifest.toml

This instructs Julia to use the Project.toml and Manifest.toml files in the current directory for managing dependencies


1) Move to the GroupIWord2Vec directory
2) Here, insert

       using Pkg
   to load the Pkg package.
4) And:

       using Pkg.activate(".")
5) Move to the package mode using:

       ]
7) Build the package:
   
       build GroupIWord2Vec
10) Leave package mode and insert:
    
        using GroupIWord2Vec

## Dependencies

## Examples
