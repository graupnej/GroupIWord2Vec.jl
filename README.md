[![Coverage](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl)
[![Build Status](https://github.com/graupnej/GroupIWord2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/graupnej/GroupIWord2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://graupnej.github.io/GroupIWord2Vec.jl/dev/)

<div align="center">
  <img src="WordEmbeddings.png" alt="Logo" width="250" height="250" />
  <h1>Word2Vec</h1>
  A Julia package that implements some tools for running and testing word-embedding algorithms
  <br/>
  <a href="https://julialang.org/downloads/">
    <img src="https://img.shields.io/badge/Julia-v1.10-blue" alt="Julia Version"/>
  </a>
</div>

### What's it about
[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks.

## Getting Started

### 1) Download
We can't use Pluto's environments but have to create our own

```julia
julia> using Pkg
julia> Pkg.activate("MyEnv")
julia> Pkg.add(url="https://github.com/graupnej/GroupIWord2Vec.jl")
julia> using GroupIWord2Vec
```

Below is an overview of the project's main components

```
Word2Vec.jl
├── src/
│   ├── GroupIWord2Vec.jl
│   ├── functions.jl
│   └── model.jl
├── test/
│   ├── runtests.jl
│   ├── test_functions.jl
│   └── test_model.jl
├── docs/
├── Manifest.toml
├── Project.toml
└── README.md
```

### 2) Running a simple example
For a simple example use http://mattmahoney.net/dc/text8.zip as text corpus to train the model. Store this file in the current working directory

To train the model based on ``text8`` use the function ``train_model``

```julia
julia> train_model("text8", "text8.txt", verbose = true)
```

The resultung word vectors are saved in a text format file.

- Note that this function currently interfaces with C code and is therefore not pure Julia. This will be updated asap.

In the next step the obtained word vectors in ``text8.txt`` can be imported to Julia.

```julia
julia> model = load_embeddings("./text8.txt")
```

Further, the package includes the following functions

- Get the vector representation of a word (``get_vector``)

```julia
julia> get_vector_from_word(model, "king")
```
```julia
100-element Vector{Float64}:
 -0.0915921031903591
 -0.10155618557541449
  0.05258880267427831
  ⋮
 -0.05509991571538997
 -0.06181055625996383
 -0.08482664361123718
```

- Get the cosine similarity of two words (``cosine_similarity``)

```julia
julia> cosine_similarity(model, "king", "prince")
```

- Get the top-n most similar words to a given word (``get_similarity``)

```julia
julia> get_similarity(model, "king", 5)
```

- Display one of the similar words

```julia
julia> model.words[1062]
```

- Plot the top-n most similar words

```
Work in progress
```

## 3) Running a large example
Use a text corpus from e.g. FastText (.bin & .vec file) https://fasttext.cc/docs/en/pretrained-vectors.html. This file includes about 33049795 words in the training file.
Store this file in the current working directory and use the same functions presented in the simple example.

# How to run tests
For code coverage we have implemented testing routines. To execute the tests, type in your Julia REPL

```julia
julia> Pkg.test("GroupIWord2Vec")
```

This covers all the tests. To execute a specific test (example), type in your Julia REPL

```julia
julia> Pkg.test("GroupIWord2Vec", test_args=["Functions"])
```

# Downloading the code
To not only use but also work on the code yourself download the repository by typing

``` bash
git clone https://github.com/graupnej/GroupIWord2Vec.jl.git
```

Navigate to the cloned directory

``` bash
cd GroupIWord2Vec.jl
```

Launch julia from the directory and activate the project environment (this tells julia to use the Project.toml)

``` bash
julia
```
```julia
julia> using Pkg
julia> Pkg.activate(".")
```

Run the following command to resolve dependencies and create a Manifest.toml file

```julia
julia> Pkg.instantiate()
```

Precompile the project to ensure all dependencies and your code are ready

```julia
julia> Pkg.precompile()
```

Run the tests to verify everything is working

```julia
julia> Pkg.test()
```

# Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra
       Plots                 # For visualization functions
       Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)

The files Project.toml and Manifest.toml in the created environment manage dependencies.

# References
The text corpus for the simple example (``text8``) is a preprocessed version of the first 100 million bytes of the English Wikipedia dump from March 3, 2006. It has been filtered to include only lowercase letters (a–z) and spaces, reducing the dataset's size to approximately 100 MB. It is commonly used for training and evaluating language models.

The text corpus for the large example were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.

       P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

As inspiration on how to properly develop julia packages, organize source files, write meaningful tests and more read [here](https://adrianhill.de/julia-ml-course/write/).

## Contributors
[![Contributors](https://contrib.rocks/image?repo=graupnej/GroupIWord2Vec)](https://github.com/graupnej/GroupIWord2Vec.jl/graphs/contributors)
