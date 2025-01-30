var documenterSearchIndex = {"docs":
[{"location":"functions/","page":"Functions","title":"Functions","text":"Modules = [GroupIWord2Vec]","category":"page"},{"location":"functions/#GroupIWord2Vec.GroupIWord2Vec","page":"Functions","title":"GroupIWord2Vec.GroupIWord2Vec","text":"GroupIWord2Vec\n\nThis is the main module file that organizes all the word embedding functionality.\n\nTypes\n\nWordEmbedding: Main data structure for word embeddings\n\nFunctions\n\ntrain_model:             Train new word embeddings\nload_embeddings:         Load pre-trained embeddings\nget_word2vec:            Function to get a word's embedded vector\nget_vec2word:            Function to get a vector's word\nget_any2vec:             Function to handle word/ vector input and convert input word/ vector into vector\nget_cosine_similarity:   Function to compute similarity of two words\nget_similar_words:       Function to find top n similar words as strings\nget_vector_operation:    Function to find perform operation on 2 input words/vectors: sum, subtract, dot-product, euclidean distance\nget_word_analogy:        Function to use vector math to compute analogous words\nshow_relations:          Function to visualise vectors and vector distances in 2D\n\n\n\n\n\n","category":"module"},{"location":"functions/#GroupIWord2Vec.WordEmbedding","page":"Functions","title":"GroupIWord2Vec.WordEmbedding","text":"WordEmbedding{S<:AbstractString, T<:Real}\n\nA structure for storing and managing word embeddings, where each word is associated with a vector representation.\n\nFields\n\nwords::Vector{S}: List of all words in the vocabulary\nembeddings::Matrix{T}: Matrix where each column is a word's vector representation\nword_indices::Dict{S, Int}: Dictionary mapping words to their positions in the vocabulary\n\nType Parameters\n\nS: Type of strings used (defaults to String)\nT: Type of numbers in the embedding vectors (defaults to Float64)\n\nConstructor\n\nWordEmbedding(words::Vector{S}, matrix::Matrix{T}) where {S<:AbstractString, T<:Real}\n\nCreates a WordEmbedding with the given vocabulary and corresponding vectors.\n\nArguments\n\nwords::Vector{S}: Vector of words in the vocabulary\nmatrix::Matrix{T}: Matrix where each column corresponds to one word's vector\n\nThrows\n\nArgumentError: If the number of words doesn't match the number of vectors (matrix columns)\n\nExample\n\n```julia\n\nCreate a simple word embedding with 2D vectors\n\nwords = [\"cat\", \"dog\", \"house\"] vectors = [0.5 0.1 0.8;           0.2 0.9 0.3] embedding = WordEmbedding(words, vectors)\n\n\n\n\n\n","category":"type"},{"location":"functions/#GroupIWord2Vec.get_any2vec-Tuple{WordEmbedding, Any}","page":"Functions","title":"GroupIWord2Vec.get_any2vec","text":"get_any2vec(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}) -> Vector{Float64}\n\nConverts a word into its corresponding vector or returns the vector unchanged if already provided. This allows other functions to take both words and vectors as input.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\nword_or_vec::Union{String, Vector{Float64}}: A word or an embedding vector.\n\nReturns\n\nVector{Float64}: The corresponding embedding vector.\n\nExample 1\n\nbanana_vec = get_any2vec(wv, \"banana\")\n\n#Example 2\n\nbanana_vec = get_any2vec(wv, banana_vec)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_similar_words","page":"Functions","title":"GroupIWord2Vec.get_similar_words","text":"get_similar_words(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}, n::Int=10) -> Vector{String}\n\nFinds the top n most similar words to a given word or vector.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\nword_or_vec::Union{String, Vector{Float64}}: A word or an embedding vector.\nn::Int: Number of similar words to return (default: 10).\n\nReturns\n\nVector{String}: List of most similar words.\n\nExample\n\nsimilar_words = get_similar_words(wv, \"king\", 5)\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.get_vec2word-Tuple{WordEmbedding, Vector{Float64}}","page":"Functions","title":"GroupIWord2Vec.get_vec2word","text":"get_vec2word(wv::WordEmbedding, vec::Vector{Float64}) -> String\n\nRetrieves the closest word in the embedding space to a given vector.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\nvec::Vector{Float64}: The embedding vector.\n\nReturns\n\nString: The word closest to the given vector.\n\nExample\n\nword = get_vec2word(wv, some_vector)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_vector_operation-Tuple{WordEmbedding, Union{String, Vector{Float64}}, Union{String, Vector{Float64}}, String}","page":"Functions","title":"GroupIWord2Vec.get_vector_operation","text":"get_vector_operation(wv::WordEmbedding, word_or_vec::Union{String, Vector{Float64}}, operator::String} -> Vector{Float64}, Float64\n\nFinds the top n most similar words to a given word or vector.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\ninp1::Union{String, Vector{Float64}}: First word or vector.\ninp2::Union{String, Vector{Float64}}: Second word or vector.\noperator::String: The operator string to define the calculation.\n\noperators can be: \"+\" -> sum, \"-\" -> subtraction, \"*\" -> dot product/ cosine similarity, \"euclid\" -> Euclidean distance\n\nReturns\n\nVector{Float64}: For operations with vecctorial result: '+' and '-'\nFloat64: For operations with scalar result: '*' and 'euclid'\n\nExample\n\nsimilar_words = get_similar_words(wv, \"king\", 5)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_word2vec-Tuple{WordEmbedding, String}","page":"Functions","title":"GroupIWord2Vec.get_word2vec","text":"get_word2vec(wv::WordEmbedding, word::String) -> Vector{Float64}\n\nRetrieves the embedding vector corresponding to a given word.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\nword::String: The word to look up.\n\nReturns\n\nVector{Float64}: The embedding vector corresponding to the word.\n\nExample\n\nvec = get_word2vec(wv, \"apple\")\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_word_analogy","page":"Functions","title":"GroupIWord2Vec.get_word_analogy","text":"get_word_analogy(wv::WordEmbedding, inp1::Union{String, Vector{Float64}}, inp2::Union{String, Vector{Float64}}, inp3::Union{String, Vector{Float64}}, n::Int=5) -> Vector{String}\n\nPerforms word analogy calculations like: king - queen + woman = man.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\ninp1::Union{String, Vector{Float64}}: First input (e.g., \"king\", vectorembeddingking).\ninp2::Union{String, Vector{Float64}}: Second input (e.g., \"queen\", vectorembeddingqueen).\ninp3::Union{String, Vector{Float64}}: Third input (e.g., \"woman\", vectorembeddingwoman).\nn::Int: Number of similar words to return (default: 5).\n\nReturns\n\nVector{String}: List of most similar words to the resulting vector.\n\nExample\n\nanalogy_result = get_word_analogy(wv, \"king\", \"queen\", \"woman\")\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.read_binary_format-Union{Tuple{T}, Tuple{AbstractString, Type{T}, Bool, Char, Int64}} where T<:Real","page":"Functions","title":"GroupIWord2Vec.read_binary_format","text":"This function reads word embeddings (word->vector mappings) from a binary file\n\nIt requires the following Parameters:\n\nfilepath: where the file is located\n\nT: what kind of numbers we want (like decimal numbers)\n\nnormalize: whether to make all vectors have length 1\n\n–-> This can be useful for comparison since the length of the vector does not\n\nmatter, only its direction\n\nseparator: what character separates the values in the file (like space or comma)\n\nskip_bytes: how many bytes to skip after each word-vector pair (usually for handling separators)\n\nInstead of reading lines of text and parsing numbers it reads words until it hits a separator\n\nReads raw bytes and converts them directly to numbers\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.read_text_format-Union{Tuple{T}, Tuple{AbstractString, Type{T}, Bool, Char}} where T<:Real","page":"Functions","title":"GroupIWord2Vec.read_text_format","text":"This function reads word embeddings (word->vector mappings) from a text file\n\nIt requires the following Parameters:\n\nfilepath: where the file is located\n\nT: what kind of numbers we want (like decimal numbers)\n\nnormalize: whether to make all vectors have length 1\n\n–-> This can be useful for comparison since the length of the vector does not\n\nmatter, only its direction\n\nseparator: what character separates the values in the file (like space or comma)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.reduce_to_2d","page":"Functions","title":"GroupIWord2Vec.reduce_to_2d","text":"reduce_to_2d(data::Matrix, number_of_pc::Int=2) -> Matrix{Float64}\n\nPerforms Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset NxM to Nx\"numberofpc\" and returns a projected data\n\nArguments\n\ndata::Matrix: The input data matrix where rows represent samples and columns represent features.\nnumber_of_pc::Int=2: The number of principal components to retain (default: 2).\n\nReturns\n\nMatrix{Float64}: A matrix of shape (number_of_pc × N), where N is the number of samples, containing the projected data in the reduced dimensional space.\n\nExample\n\n```julia data = randn(100, 50)  # 100 samples, 50 features reduceddata = reduceto_2d(data, 2)\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.show_relations-Tuple{Vararg{String}}","page":"Functions","title":"GroupIWord2Vec.show_relations","text":"show_relations(words::String...; wv::WordEmbedding, save_path::String=\"word_relations.png\") -> Plots.Plot\n\nGenerates a 2D PCA projection of the given word embeddings and visualizes their relationships like this: arg1==>arg2, arg3==>arg4, ... Note: Use an even number of inputs!\n\nArguments\n\nwords::String...: A list of words to visualize. The number of words must be a multiple of 2.\nwv::WordEmbedding: The word embedding model containing the word vectors.\nsave_path::String=\"word_relations.png\": The file path where the generated plot will be saved. If empty or nothing, the plot is not saved.\n\nThrows\n\nArgumentError: If the number of words is not a multiple of 2.\nArgumentError: If any of the provided words are not found in the embedding model.\n\nReturns\n\nPlots.Plot: A scatter plot with arrows representing word relationships.\n\nExample\n\n```julia p = showrelations(\"king\", \"queen\", \"man\", \"woman\"; wv=model, savepath=\"relations.png\")\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.train_model-Tuple{AbstractString, AbstractString}","page":"Functions","title":"GroupIWord2Vec.train_model","text":" word2vec(train, output; size=100, window=5, sample=1e-3, hs=0,  negative=5, threads=12, iter=5, min_count=5, alpha=0.025, debug=2, binary=1, cbow=1, save_vocal=Nothing(), read_vocab=Nothing(), verbose=false,)\n\nParameters for training:\n    train <file>\n        Use text data from <file> to train the model\n    output <file>\n        Use <file> to save the resulting word vectors / word clusters\n    size <Int>\n        Set size of word vectors; default is 100\n    window <Int>\n        Set max skip length between words; default is 5\n    sample <AbstractFloat>\n        Set threshold for occurrence of words. Those that appear with\n        higher frequency in the training data will be randomly\n        down-sampled; default is 1e-5.\n    hs <Int>\n        Use Hierarchical Softmax; default is 1 (0 = not used)\n    negative <Int>\n        Number of negative examples; default is 0, common values are \n        5 - 10 (0 = not used)\n    threads <Int>\n        Use <Int> threads (default 12)\n    iter <Int>\n        Run more training iterations (default 5)\n    min_count <Int>\n        This will discard words that appear less than <Int> times; default\n        is 5\n    alpha <AbstractFloat>\n        Set the starting learning rate; default is 0.025\n    debug <Int>\n        Set the debug mode (default = 2 = more info during training)\n    binary <Int>\n        Save the resulting vectors in binary moded; default is 0 (off)\n    cbow <Int>\n        Use the continuous back of words model; default is 1 (skip-gram\n        model)\n    save_vocab <file>\n        The vocabulary will be saved to <file>\n    read_vocab <file>\n        The vocabulary will be read from <file>, not constructed from the\n        training data\n    verbose <Bool>\n        Print output from training\n\n\n\n\n\n","category":"method"},{"location":"getting_started/#Download-For-users","page":"Getting Started","title":"Download - For users","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We can't use Pluto's environments but have to create our own","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> using Pkg\njulia> Pkg.activate(\"MyEnv\")\njulia> Pkg.add(url=\"https://github.com/graupnej/GroupIWord2Vec.jl\")\njulia> using GroupIWord2Vec","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Below is an overview of the project's main components","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"GroupIWord2Vec.jl               \n├── src/                        # Contains core modules for the package\n│   ├── GroupIWord2Vec.jl       # Main entry point for the project\n│   ├── functions.jl            # Word/vector functions\n│   └── model.jl                # Model functions\n├── test/                       # Unit tests to validate functionalities\n│   ├── runtests.jl             # Combination of every testing routine\n│   ├── test_functions.jl       # Testing routine for word/vector functions \n│   └── test_model.jl           # Testing routine for model functions\n├── docs/                       # Documentation for the package\n├── Manifest.toml               # Detailed dependency lock file that tracks exact versions of project dependencies\n├── Project.toml                # Project configuration file defining package dependencies\n└── README.md                   # Main documentation file containing getting started","category":"page"},{"location":"getting_started/#Running-a-simple-example","page":"Getting Started","title":"Running a simple example","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Download https://mattmahoney.net/dc/text8.zip and store it in the current working directory. To train the model with text8 use train_model()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> train_model(\"text8\", \"text8.txt\", verbose = true)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The resulting word vectors are saved in a text format file (here) named text8txt. Import the obtained word vectors from text8txt into Julia using load_embeddings()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> model = load_embeddings(\"./text8.txt\")","category":"page"},{"location":"getting_started/#Some-functionalities","page":"Getting Started","title":"Some functionalities","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get _vector _from _word(): Get the vector representation of a word","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_vector_from_word(model, \"king\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"cosine_similarity(): Returns cosine of the angle between two vectors in a word embedding space","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> cosine_similarity(model, \"king\", \"prince\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"It ranges from -1 to 1, where 1 indicates high similarity, 0 indicates no similarity and -1 indicates opposite directions.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get _top _similarity _of _word(): Find the n most similar words to a given word and return the matching strings","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_top_similarity_of_word(model, \"king\", 5)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"word_analogy(): Performs word analogy calculations (e.g. king - man + woman = queen)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> word_analogy(model, [\"king\", \"woman\"], [\"man\"])","category":"page"},{"location":"getting_started/#Display-Data","page":"Getting Started","title":"Display Data","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"show_relations(): Creates a PCA Projection to 2D of words with connecting vectors ","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> show_relations(\"berlin\", \"germany\", \"paris\", \"france\", \"rome\", \"apple\", wv=model, save_path=\"my_custom_plot.png\")","category":"page"},{"location":"getting_started/#Running-a-large-example","page":"Getting Started","title":"Running a large example","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"As an alternative (larger) example use a text corpus from e.g. FastText (.bin & .vec file) https://fasttext.cc/docs/en/pretrained-vectors.html with about 33 million words. Store this file in the current working directory and apply the same functions as in the previous example.","category":"page"},{"location":"getting_started/#Download-For-Developers","page":"Getting Started","title":"Download - For Developers","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Clone the code using","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"git clone https://github.com/graupnej/GroupIWord2Vec.jl.git","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Navigate to the cloned directory and launch julia. Activate the project environment to tell Julia to use the Project.toml","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> using Pkg\njulia> Pkg.activate(\".\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Resolve dependencies and create Manifest.toml file","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.instantiate()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Precompile the project to ensure all dependencies and the code is ready","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.precompile()","category":"page"},{"location":"getting_started/#Run-tests","page":"Getting Started","title":"Run tests","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"To verify everything is working correctly run the code coverage tests","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.test(\"GroupIWord2Vec\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"This covers all the tests. To execute a specific test (e.g. Functions)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.test(\"GroupIWord2Vec\", test_args=[\"Functions\"])","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GroupIWord2Vec","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Coverage) (Image: Build Status) (Image: Dev)","category":"page"},{"location":"#Explanation","page":"Home","title":"Explanation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Word Embeddings are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks.","category":"page"},{"location":"#Prerequisities-and-dependencies","page":"Home","title":"Prerequisities and dependencies","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package uses (Image: Julia) and relies on the following non-standard Julia packages:","category":"page"},{"location":"","page":"Home","title":"Home","text":"   DelimitedFiles        # Provides functionality for reading and writing delimited text files\n   LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra\n   Plots                 # For visualization functions\n   Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)\n   Statistics            # For basic statistical operations (mean, std, var, etc.)","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
