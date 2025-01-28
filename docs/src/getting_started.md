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
GroupIWord2Vec.jl               
├── src/                        # Contains core modules for the package
│   ├── GroupIWord2Vec.jl       # Main entry point for the project
│   ├── functions.jl            # Word/vector functions
│   └── model.jl                # Model functions
├── test/                       # Unit tests to validate functionalities
│   ├── runtests.jl             # Combination of every testing routine
│   ├── test_functions.jl       # Testing routine for word/vector functions 
│   └── test_model.jl           # Testing routine for model functions
├── docs/                       # Documentation for the package
├── Manifest.toml               # Detailed dependency lock file that tracks exact versions of project dependencies
├── Project.toml                # Project configuration file defining package dependencies
└── README.md                   # Main documentation file containing getting started
```

### 2) Running a simple example
Download https://mattmahoney.net/dc/text8.zip and store it in the current working directory. To train the model with text8 use ``train``_``model()``

```julia
julia> train_model("text8", "text8.txt", verbose = true)
```

The resulting word vectors are saved in a text format file (here) named ``text8.txt``.
Import the obtained word vectors from ``text8.txt`` into Julia using ``load_embeddings()``

```julia
julia> model = load_embeddings("./text8.txt")
```

#### Some functionalities

- ``get_vector_from_word()``: Get the vector representation of a word

```julia
julia> get_vector_from_word(model, "king")
```



- ``cosine_similarity()``: Returns cosine of the angle between two vectors in a word embedding space

```julia
julia> cosine_similarity(model, "king", "prince")
```

It ranges from -1 to 1, where 1 indicates high similarity, 0 indicates no similarity and -1 indicates opposite directions.
 
- ``get_top_similarity_of_word()``: Find the n most similar words to a given word and return the matching strings

```julia
julia> get_top_similarity_of_word(model, "king", 5)
```

- ``word_analogy()``: Performs word analogy calculations (e.g. king - man + woman = queen)
  
```julia
julia> word_analogy(model, ["king", "woman"], ["man"])
```

### 3) Running a large example
As an alternative (larger) example use a text corpus from e.g. FastText (.bin & .vec file) https://fasttext.cc/docs/en/pretrained-vectors.html with about 33 million words. Store this file in the current working directory and apply the same functions as in the previous example.

## For Developers
### 1) Download the code

``` bash
git clone https://github.com/graupnej/GroupIWord2Vec.jl.git
```

Navigate to the cloned directory and launch julia. Activate the project environment to tell Julia to use the Project.toml

```julia
julia> using Pkg
julia> Pkg.activate(".")
```

Resolve dependencies and create a Manifest.toml file

```julia
julia> Pkg.instantiate()
```

Precompile the project to ensure all dependencies and the code is ready

```julia
julia> Pkg.precompile()
```

### 2) Run tests
To verify everything is working correctly run the code coverage tests

```julia
julia> Pkg.test("GroupIWord2Vec")
```

This covers all the tests. To execute a specific test (e.g. Functions)

```julia
julia> Pkg.test("GroupIWord2Vec", test_args=["Functions"])
```

