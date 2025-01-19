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
Via `Pkg>` mode (press `]` in Julia REPL):

```bash
add https://github.com/graupnej/GroupIWord2Vec.jl
```

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
Starting training using file text8
Vocab size: 71291
Words in train file: 16718843
Alpha: 0.000002  Progress: 100.01%  Words/thread/sec: 159.38k  
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
julia> get_vector(model, "book")'
100-element Vector{Float64}:
 -0.0915921031903591
 -0.10155618557541449
  0.05258880267427831
  ⋮
 -0.05509991571538997
 -0.06181055625996383
 -0.08482664361123718
```

- Get the top-n most similar words to a given word (``get_similar``)

```
Will be implemented soon
```

# Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra

# References
The text corpus for the simple example (``text8``) is a preprocessed version of the first 100 million bytes of the English Wikipedia dump from March 3, 2006. It has been filtered to include only lowercase letters (a–z) and spaces, reducing the dataset's size to approximately 100 MB. It is commonly used for training and evaluating language models.

The text corpus for the complex example were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.

       P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information


## 1) Clone the github repository
First, clone the repository to your local machine (replace <username> with your details):

       git clone https://github.com/<username>/GroupIWord2Vec.jl.git

Once downloaded, move the directory into your julia directory:

       mv GroupIWord2Vec.jl ~.julia/

## 2) Open the package in Julia
Launch the Julia REPL and navigate to the cloned directory:

       cd ~.julia/GroupIWord2Vec

and activate Julia's package environment:

       using Pkg
       
Here, activate the project environment:

       Pkg.activate(".")         # Activate the local environment
       Pkg.instantiate()         # Install dependencies from the Manifest.toml

This instructs Julia to use the Project.toml and Manifest.toml files in the current directory for managing dependencies.
In Julia's package management system (Pkg mode, entered by typing ]) trigger the build process for the package:

       build GroupIWord2Vec

Within the Julia REPL, load the package into the current session

       using GroupIWord2Vec

## 3) Select and implement a pre-trained Word2Vec model e.g. FastText English (.bin & .vec file):

       https://fasttext.cc/docs/en/pretrained-vectors.html

Once downloaded, move the directory to the package directory:

       mv wiki.en ~.julia/GroupIWord2Vec

This is what the file structure should look like:

       .julia/
           └── GroupIWord2Vec/         # Development directory for the package
               ├── src/
               │   ├── GroupIWord2Vec.jl      # Main package file
               │   ├── functions.jl           # Main functions
               ├── test/
               │   ├── runtests.jl            # Test suite
               ├── wiki.en/                   # Pretrained embeddings
               │   ├── wiki.en.vec            # Vector file
               │   ├── wiki.en.bin            # Binary file
               ├── LICENSE                    # License file
               ├── .gitignore                 # Git ignore rules
               ├── Project.toml               # Package dependencies
               ├── Manifest.toml              # Dependency snapshot
               └── README.md                  # Documentation

# Dependencies
GroupIWord2Vec.jl relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra

# Examples
In the package directory:

       ~.julia/GroupIWord2Vec

Load a pre-trained model from a file in text format:

       model_vec = load_text_model("wiki.en/wiki.en.vec")

Alternatively, load a pre-trained model from a file in binary format:

       model_bin = load_text_model("wiki.en/wiki.en.bin")

Generate a word embedding for a given word using one of the loaded models:

       embedding = get_word_embedding(model_, "test")

# Run a test
In order to compare the vectors from both files you can run a test in the package directory:

       Pkg.test("GroupIWord2Vec")

This compares the vectors for certain predefined words from both files to check whether the files have been read similarly or not.

