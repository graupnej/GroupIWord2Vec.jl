# Getting Started

## 1) How to install, build and load the package
To install the package, use Julia's package manager:

       using Pkg
       Pkg.add(GroupIWord2Vec)
       
Move to the GroupIWord2Vec directory and activate a project environment in this directory:

       Pkg.activate(".")  # Activate the local environment
       Pkg.instantiate()  # Install dependencies from the Manifest.toml

This instructs Julia to use the Project.toml and Manifest.toml files in the current directory for managing dependencies.
In Julia's package management system (activate with ]) trigger the build process for the package:

       build GroupIWord2Vec

Load the package into the current session

       using GroupIWord2Vec

## 2) Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles # Provides functionality for reading and writing delimited text files
       LinearAlgebra  # Offers a suite of mathematical tools and operations for linear algebra

## 3) Examples

## 4) References
