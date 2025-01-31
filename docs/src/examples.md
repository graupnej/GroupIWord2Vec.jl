# Walkthroug Examples:

After training and loading the model a WordEmbeddings dictionary storing all words, corresponding embedding vectors and indices is given.

```@example
using GroupIWord2Vec

example_txt = "The quick brown fox jumps over the lazy dog. Every morning, the sun rises in the east and sets in the west. People gather in the park, enjoying the fresh air and vibrant atmosphere. The world spins on, offering endless opportunities for adventure, discovery, and personal growth."

example_txt_emb = []

train_model(example_txt, example_txt_emb, verbose = true)
model = load_embeddings(example_txt_emb)
```

Now that the embeddings were created we can use the functions to do some operations.


```@example
king_vec = get_word2vec(model, "king")
```

### Train Model and Create Word Embeddings - Text8

Download the text corpus [_text8_](https://mattmahoney.net/dc/text8.zip) and store it in the current working directory. To train the model with this text corpus use ``train_model()``

```julia
julia> train_model("text8", "text8.txt", verbose = true)
```

The resulting word vectors are saved in a text format file (here) named _text8.txt_.
Import the obtained word vectors from _text8.txt_ into Julia using ``load_embeddings()``

```julia
julia> model = load_embeddings("./text8.txt")
```

### Examples
#### Functions

Now that a model is loaded the functions of this package can be used to work with the embedding vectors.


- ``get_word2vec()``: Retrieves the embedding vector corresponding to a given word.

```julia
julia> get_word2vec(model, "king")
```

- ``get_vec2word()``: Retrieves the closest word in the embedding space to a given vector.

```julia
julia> get_vec2word(model, king_vec)
```

- ``get_vector_operation()``: Computes 1 of 4 vector calculations on two input words or vectors depending on the input operator

```julia
julia> get_vector_operation(model, "king", "queen",:+)
```
or
```julia
julia> get_vector_operation(model, king_vec, "queen","euclid")
```

- ``get_word_analogy()``: Performs word analogy calculations (e.g. king - man + woman = queen)
  
```julia
julia> word_analogy(model, "king", "man", "woman")
```

#### Display Data Functions
- ``show``_``relations()``: Creates a [PCA Projection](https://en.wikipedia.org/wiki/Principal_component_analysis) to 2D of words with connecting vectors 

```julia
julia> show_relations("berlin", "germany", "paris", "france", "rome", "apple", wv=model, save_path="my_custom_plot.png")
```

<div align="center">
  <img src="assets/PCAProjection.png" alt="Logo" width="400" height="250" />
</div>

### Train Model and Create Word Embeddings - fasttext
As an alternative use a (larger) text corpus from e.g. [FastText](https://fasttext.cc/docs/en/pretrained-vectors.html) (.bin & .vec file) with about 33 million words. Store this file in the current working directory and apply the same functions as in the previous example.


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
