[![codecov](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl/graph/badge.svg?token=WRWRU7P5WQ)](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl)

<div align="center">
  <img src="WordEmbeddings.png" alt="Logo" width="300" height="300" />
       <h1>Word2Vec</h1>
       A Julia package that implements some tools for running and testing word-embedding algorithms
</div>

## What's it about
[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks.

# Getting Started

## 1) Download
We can't use Pluto's environments but have to create our own

```julia
using Pkg
Pkg.activate("MyEnvironment")
Pkg.add(url="https://github.com/graupnej/GroupIWord2Vec.jl")
using GroupIWord2Vec
```

## 2) Running a simple example
In order to train a model some text corpus is required. For a simple example use http://mattmahoney.net/dc/text8.zip. Store this file in the current working directory

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
julia> get_vector(model, "king")
100-element Vector{Float64}:
 -0.0915921031903591
 -0.10155618557541449
  0.05258880267427831
  ⋮
 -0.05509991571538997
 -0.06181055625996383
 -0.08482664361123718
```

- Get the top-n most similar words to a given word (``get_similarity``)

```julia
julia> get_similarity(model, "king", 5)
([188, 1062, 904, 527, 1245], [1.0, 0.7518736087237998, 0.715927240172969, 0.6939850961445455, 0.678069618100706])
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

# Run a test
For code coverage we have implemented testing routines

       This is still in progress

# Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra
       Plots                 # For visualization functions
       Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)

The files Project.toml and Manifest.toml in the created environment manage dependencies.

# References
The text corpus for the simple example (``text8``) is a preprocessed version of the first 100 million bytes of the English Wikipedia dump from March 3, 2006. It has been filtered to include only lowercase letters (a–z) and spaces, reducing the dataset's size to approximately 100 MB. It is commonly used for training and evaluating language models.

The text corpus for the complex example were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.

       P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information

