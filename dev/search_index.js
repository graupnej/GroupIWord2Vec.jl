var documenterSearchIndex = {"docs":
[{"location":"functions/","page":"Functions","title":"Functions","text":"Modules = [GroupIWord2Vec]","category":"page"},{"location":"functions/#GroupIWord2Vec.GroupIWord2Vec","page":"Functions","title":"GroupIWord2Vec.GroupIWord2Vec","text":"GroupIWord2Vec\n\nThis is the main module file that organizes all the word embedding functionality.\n\nTypes\n\nWordEmbedding: Main data structure for word embeddings\n\nFunctions\n\ntrain_model:             Train new word embeddings\nload_embeddings:         Load pre-trained embeddings\nget_word2vec:            Function to get a word's embedded vector\nget_vec2word:            Function to get a vector's word\nget_any2vec:             Function to handle word/ vector input and convert input word/ vector into vector\nget_similar_words:       Function to find top n similar words as strings\nget_vector_operation:    Function to find perform operation on 2 input words/vectors: sum, subtract, dot-product, euclidean distance\nget_word_analogy:        Function to use vector math to compute analogous words\nshow_relations:          Function to visualise vectors and vector distances in 2D\n\n\n\n\n\n","category":"module"},{"location":"functions/#GroupIWord2Vec.WordEmbedding","page":"Functions","title":"GroupIWord2Vec.WordEmbedding","text":"WordEmbedding{S<:AbstractString, T<:Real}\n\nA structure for storing and managing word embeddings, where each word is associated with a vector representation.\n\nFields\n\nwords::Vector{S}: List of all words in the vocabulary\nembeddings::Matrix{T}: Matrix where each column is a word's vector representation\nword_indices::Dict{S, Int}: Dictionary mapping words to their positions in the vocabulary\n\nType Parameters\n\nS: Type of strings used (defaults to String)\nT: Type of numbers in the embedding vectors (defaults to Float64)\n\nConstructor\n\nWordEmbedding(words::Vector{S}, matrix::Matrix{T}) where {S<:AbstractString, T<:Real}\n\nCreates a WordEmbedding with the given vocabulary and corresponding vectors.\n\nArguments\n\nwords::Vector{S}: Vector of words in the vocabulary\nmatrix::Matrix{T}: Matrix where each column corresponds to one word's vector\n\nThrows\n\nArgumentError: If the number of words doesn't match the number of vectors (matrix columns)\n\nExample\n\n```julia\n\nCreate a simple word embedding with 2D vectors\n\nwords = [\"cat\", \"dog\", \"house\"] vectors = [0.5 0.1 0.8;           0.2 0.9 0.3] embedding = WordEmbedding(words, vectors)\n\n\n\n\n\n","category":"type"},{"location":"functions/#GroupIWord2Vec.get_any2vec-Union{Tuple{T}, Tuple{S}, Tuple{WordEmbedding{S, T}, Union{Vector{<:Real}, S}}} where {S<:AbstractString, T<:Real}","page":"Functions","title":"GroupIWord2Vec.get_any2vec","text":"get_any2vec(wv::WordEmbedding{S, T}, word_or_vec::Union{S, Vector{<:Real}}) -> Vector{T} \nwhere {S<:AbstractString, T<:Real}\n\nConverts a word into its corresponding vector representation or returns the vector unchanged if already provided.\n\nArguments\n\nwv::WordEmbedding{S, T}: A word embedding structure with words and their corresponding vector representations.\nword_or_vec::Union{S, Vector{<:Real}}: A word to be converted into a vector, or a numerical vector to be validated.\n\nReturns\n\nVector{T}: The vector representation of the word if input is a String, or the validated vector (converted to T if necessary).\n\nThrows\n\nDimensionMismatch: If the input vector does not match the embedding dimension.\nArgumentError: If the input is neither a word nor a valid numeric vector.\n\nExample\n\n```julia words = [\"cat\", \"dog\"] vectors = [0.5 0.1;           0.2 0.9] wv = WordEmbedding(words, vectors)\n\ngetany2vec(wv, \"cat\")  # Returns [0.5, 0.2] getany2vec(wv, [0.5, 0.2])  # Returns [0.5, 0.2]\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_similar_words","page":"Functions","title":"GroupIWord2Vec.get_similar_words","text":"get_similar_words(wv::WordEmbedding, word_or_vec::Union{AbstractString, AbstractVector{<:Real}}, n::Int=10) -> Vector{String}\n\nFinds the top n most similar words to a given word or vector based on cosine similarity.\n\nArguments\n\nwv::WordEmbedding: The word embedding model containing the vocabulary and embeddings.\nword_or_vec::Union{AbstractString, AbstractVector{<:Real}}: The target word or embedding vector.\nn::Int=10: The number of most similar words to retrieve (default is 10).\n\nThrows\n\nArgumentError: If the input word is not found in the vocabulary and is not a valid vector.\nDimensionMismatch: If the input vector does not match the embedding dimension.\nArgumentError: If the input vector has zero norm, making similarity computation invalid.\n\nReturns\n\nVector{String}: A list of the n most similar words ordered by similarity score.\n\nExample\n\n```julia similarwords = getsimilar_words(model, \"cat\", 5)\n\nExample output: [\"dog\", \"kitten\", \"feline\", \"puppy\", \"pet\"]\n\nvec = getword2vec(model, \"ocean\") similarwords = getsimilarwords(model, vec, 3)\n\nExample output: [\"sea\", \"water\", \"wave\"]\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.get_vec2word-Union{Tuple{T}, Tuple{S}, Tuple{WordEmbedding{S, T}, Vector{T}}} where {S<:AbstractString, T<:Real}","page":"Functions","title":"GroupIWord2Vec.get_vec2word","text":"get_vec2word(wv::WordEmbedding{S, T}, vec::Vector{T}) where {S<:AbstractString, T<:Real} -> String\n\nRetrieves the closest word in the embedding space to a given vector based on cosine similarity.\n\nArguments\n\nwv::WordEmbedding{S, T}: A word embedding structure with words and their corresponding vector representations.\nvec::Vector{T}: A vector representation of a word.\n\nReturns\n\nS: The word from the vocabulary closest to the given vector\n\nThrows\n\nDimensionMismatch: If the input vector's dimension does not match the word vector dimensions.\n\nExample\n\n```julia words = [\"cat\", \"dog\"] vectors = [0.5 0.1;           0.2 0.9] embedding = WordEmbedding(words, vectors)\n\nget_vec2word(embedding, [0.51, 0.19])  # Returns \"cat\"\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_vector_operation-Tuple{WordEmbedding, Union{String, AbstractVector{<:Real}}, Union{String, AbstractVector{<:Real}}, Symbol}","page":"Functions","title":"GroupIWord2Vec.get_vector_operation","text":"get_vector_operation(ww::WordEmbedding, inp1::Union{String, AbstractVector{<:Real}}, \n                     inp2::Union{String, AbstractVector{<:Real}}, operator::Symbol) -> Union{Vector{<:Real}, Float64}\n\nPerforms a mathematical operation between two word embedding vectors.\n\nArguments\n\nww::WordEmbedding: The word embedding model containing the vocabulary and embeddings.\ninp1::Union{String, AbstractVector{<:Real}}: The first input, which can be a word (String) or a precomputed embedding vector.\ninp2::Union{String, AbstractVector{<:Real}}: The second input, which can be a word (String) or a precomputed embedding vector.\noperator::Symbol: The operation to perform. Must be one of :+, :-, :cosine, or :euclid.\n\nThrows\n\nArgumentError: If the operator is invalid.\nArgumentError: If cosine similarity is attempted on a zero vector.\nDimensionMismatch: If the input vectors do not have the same length.\n\nReturns\n\nVector{<:Real}: If the operation is :+ (addition) or :- (subtraction), returns the resulting word vector.\nFloat64: If the operation is :cosine (cosine similarity) or :euclid (Euclidean distance), returns a scalar value.\n\nExample\n\n```julia vec = getvectoroperation(model, \"king\", \"man\", :-) similarity = getvectoroperation(model, \"cat\", \"dog\", :cosine) distance = getvectoroperation(model, \"car\", \"bicycle\", :euclid)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_word2vec-Tuple{WordEmbedding, String}","page":"Functions","title":"GroupIWord2Vec.get_word2vec","text":"get_word2vec(wv::WordEmbedding, word::String) -> Vector{T}\n\nRetrieves the embedding vector corresponding to a given word.\n\nArguments\n\nwv::WordEmbedding: The word embedding model containing the vocabulary and embeddings.\nword::String: The word to look up\n\nThrows\n\nArgumentError: If the word is not found in the embedding model.\n\nReturns\n\nVector{T}: The embedding vector of the requested word, where T is the numerical type of the embeddings.\n\nExample\n\n```julia vec = get_word2vec(model, \"dog\")\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.get_word_analogy","page":"Functions","title":"GroupIWord2Vec.get_word_analogy","text":"get_word_analogy(wv::WordEmbedding, inp1::T, inp2::T, inp3::T, n::Int=5) where {T<:Union{AbstractString, AbstractVector{<:Real}}} -> Vector{String}\n\nFinds the top n words that best complete the analogy: inp1 - inp2 + inp3 = ?.\n\nArguments\n\nwv::WordEmbedding: The word embedding model.\ninp1, inp2, inp3::T: Words or vectors for analogy computation.\nn::Int=5: Number of closest matching words to return.\n\nReturns\n\nVector{String}: A list of the top n matching words.\n\nNotes\n\nInput words are converted to vectors automatically.\nThe computed analogy vector is normalized.\nInput words (if given as strings) are excluded from results.\n\nExample\n\n```julia getwordanalogy(model, \"king\", \"man\", \"woman\", 3) \n\n→ [\"queen\", \"princess\", \"duchess\"]\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.load_embeddings-Union{Tuple{AbstractString}, Tuple{T}} where T<:Real","page":"Functions","title":"GroupIWord2Vec.load_embeddings","text":"load_embeddings(path::AbstractString; format::Union{:text, :binary}=:text, \n                data_type::Type{T}=Float64, normalize_vectors::Bool=true, \n                separator::Char=' ', skip_bytes::Int=0) -> WordEmbedding\n\nLoads word embeddings from a text or binary file.\n\nArguments\n\npath::AbstractString: Path to the embedding file.\nformat::Union{:text, :binary}=:text: File format (:text or :binary).\ndata_type::Type{T}=Float64: Type of word vectors (Float32, Float64, etc.).\nnormalize_vectors::Bool=true: Normalize vectors to unit length.\nseparator::Char=' ': Word-vector separator in text files.\nskip_bytes::Int=0: Bytes to skip after each word-vector pair in binary files.\n\nThrows\n\nArgumentError: If format is not :text or :binary.\n\nReturns\n\nWordEmbedding{S, T}: The loaded word embeddings.\n\nExample\n\n```julia embedding = loadembeddings(\"vectors.txt\")  # Load text format embedding = loadembeddings(\"vectors.bin\", format=:binary, datatype=Float32, skipbytes=1)  # Load binary format\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.read_binary_format-Union{Tuple{T}, Tuple{AbstractString, Type{T}, Bool, Char, Int64}} where T<:Real","page":"Functions","title":"GroupIWord2Vec.read_binary_format","text":"This function reads word embeddings (word->vector mappings) from a binary file\n\nIt requires the following Parameters:\n\nfilepath: where the file is located\n\nT: what kind of numbers we want (like decimal numbers)\n\nnormalize: whether to make all vectors have length 1\n\n–-> This can be useful for comparison since the length of the vector does not\n\nmatter, only its direction\n\nseparator: what character separates the values in the file (like space or comma)\n\nskip_bytes: how many bytes to skip after each word-vector pair (usually for handling separators)\n\nInstead of reading lines of text and parsing numbers it reads words until it hits a separator\n\nReads raw bytes and converts them directly to numbers\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.read_text_format-Union{Tuple{T}, Tuple{AbstractString, Type{T}, Bool, Char}} where T<:Real","page":"Functions","title":"GroupIWord2Vec.read_text_format","text":"This function reads word embeddings (word->vector mappings) from a text file\n\nIt requires the following Parameters:\n\nfilepath: where the file is located\n\nT: what kind of numbers we want (like decimal numbers)\n\nnormalize: whether to make all vectors have length 1\n\n–-> This can be useful for comparison since the length of the vector does not\n\nmatter, only its direction\n\nseparator: what character separates the values in the file (like space or comma)\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.reduce_to_2d","page":"Functions","title":"GroupIWord2Vec.reduce_to_2d","text":"reduce_to_2d(data::Matrix, number_of_pc::Int=2) -> Matrix{Float64}\n\nPerforms Principal Component Analysis (PCA) to reduce the dimensionality of a given dataset NxM to Nx\"numberofpc\" and returns a projected data\n\nArguments\n\ndata::Matrix: The input data matrix where rows represent samples and columns represent features.\nnumber_of_pc::Int=2: The number of principal components to retain (default: 2).\n\nReturns\n\nMatrix{Float64}: A matrix of shape (number_of_pc × N), where N is the number of samples, containing the projected data in the reduced dimensional space.\n\nExample\n\n```julia data = randn(100, 50)  # 100 samples, 50 features reduceddata = reduceto_2d(data, 2)\n\n\n\n\n\n","category":"function"},{"location":"functions/#GroupIWord2Vec.show_relations-Tuple{Vararg{String}}","page":"Functions","title":"GroupIWord2Vec.show_relations","text":"show_relations(words::String...; wv::WordEmbedding, save_path::String=\"word_relations.png\") -> Plots.Plot\n\nGenerates a 2D PCA projection of the given word embeddings and visualizes their relationships like this: arg1==>arg2, arg3==>arg4, ... Note: Use an even number of inputs!\n\nArguments\n\nwords::String...: A list of words to visualize. The number of words must be a multiple of 2.\nwv::WordEmbedding: The word embedding model containing the word vectors.\nsave_path::String=\"word_relations.png\": The file path where the generated plot will be saved. If empty or nothing, the plot is not saved.\n\nThrows\n\nArgumentError: If the number of words is not a multiple of 2.\nArgumentError: If any of the provided words are not found in the embedding model.\n\nReturns\n\nPlots.Plot: A scatter plot with arrows representing word relationships.\n\nExample\n\n```julia p = showrelations(\"king\", \"queen\", \"man\", \"woman\"; wv=model, savepath=\"relations.png\")\n\n\n\n\n\n","category":"method"},{"location":"functions/#GroupIWord2Vec.train_model-Tuple{AbstractString, AbstractString}","page":"Functions","title":"GroupIWord2Vec.train_model","text":" word2vec(train, output; size=100, window=5, sample=1e-3, hs=0,  negative=5, threads=12, iter=5, min_count=5, alpha=0.025, debug=2, binary=1, cbow=1, save_vocal=Nothing(), read_vocab=Nothing(), verbose=false,)\n\nParameters for training:\n    train <file>\n        Use text data from <file> to train the model\n    output <file>\n        Use <file> to save the resulting word vectors / word clusters\n    size <Int>\n        Set size of word vectors; default is 100\n    window <Int>\n        Set max skip length between words; default is 5\n    sample <AbstractFloat>\n        Set threshold for occurrence of words. Those that appear with\n        higher frequency in the training data will be randomly\n        down-sampled; default is 1e-5.\n    hs <Int>\n        Use Hierarchical Softmax; default is 1 (0 = not used)\n    negative <Int>\n        Number of negative examples; default is 0, common values are \n        5 - 10 (0 = not used)\n    threads <Int>\n        Use <Int> threads (default 12)\n    iter <Int>\n        Run more training iterations (default 5)\n    min_count <Int>\n        This will discard words that appear less than <Int> times; default\n        is 5\n    alpha <AbstractFloat>\n        Set the starting learning rate; default is 0.025\n    debug <Int>\n        Set the debug mode (default = 2 = more info during training)\n    binary <Int>\n        Save the resulting vectors in binary moded; default is 0 (off)\n    cbow <Int>\n        Use the continuous back of words model; default is 1 (skip-gram\n        model)\n    save_vocab <file>\n        The vocabulary will be saved to <file>\n    read_vocab <file>\n        The vocabulary will be read from <file>, not constructed from the\n        training data\n    verbose <Bool>\n        Print output from training\n\n\n\n\n\n","category":"method"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"(Image: Coverage) (Image: Build Status) (Image: Dev)","category":"page"},{"location":"getting_started/#Word-Embeddings:","page":"Getting Started","title":"Word Embeddings:","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Word Embeddings are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks. This package allows to train a ML model to create word embeddings based on a source text, and provides functionality to work with the create word embedding vectors.","category":"page"},{"location":"getting_started/#Getting-Started","page":"Getting Started","title":"Getting Started","text":"","category":"section"},{"location":"getting_started/#1)-Download","page":"Getting Started","title":"1) Download","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"We can't use Pluto's environments but have to create our own","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> using Pkg\njulia> Pkg.activate(\"MyEnv\")\njulia> Pkg.add(url=\"https://github.com/graupnej/GroupIWord2Vec.jl\")\njulia> using GroupIWord2Vec","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Below is an overview of the project's main components","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"GroupIWord2Vec.jl               \n├── src/                        # Contains core modules for the package\n│   ├── GroupIWord2Vec.jl       # Main entry point for the project\n│   ├── functions.jl            # Word/vector functions\n│   └── model.jl                # Model functions\n├── test/                       # Unit tests to validate functionalities\n│   ├── runtests.jl             # Combination of every testing routine\n│   ├── test_functions.jl       # Testing routine for word/vector functions \n│   └── test_model.jl           # Testing routine for model functions\n├── docs/                       # Documentation for the package\n├── Manifest.toml               # Detailed dependency lock file that tracks exact versions of project dependencies\n├── Project.toml                # Project configuration file defining package dependencies\n└── README.md                   # Main documentation file containing getting started","category":"page"},{"location":"getting_started/#2)-Running-a-simple-example","page":"Getting Started","title":"2) Running a simple example","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Download text8 and store it in the current working directory. To train the model with this text corpus use train_model()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> train_model(\"text8\", \"text8.txt\", verbose = true)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The resulting word vectors are saved in a text format file (here) named text8.txt. Import the obtained word vectors from text8.txt into Julia using load_embeddings()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> model = load_embeddings(\"./text8.txt\")","category":"page"},{"location":"getting_started/#Some-functionalities","page":"Getting Started","title":"Some functionalities","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_word2vec(): Retrieves the embedding vector corresponding to a given word.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_word2vec(model, \"king\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_word2vec(model, \"king\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_vec2word(): Retrieves the closest word in the embedding space to a given vector.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_vec2word(model, king_vec)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_any2vec(): Converts a word into its corresponding vector or returns the vector unchanged if already provided. This allows other functions to take both words and vectors as input.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_any2vec(model, \"king\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"or","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_any2vec(model, king_vec)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_cosine_similarity(): Returns cosine of the angle between two vectors in a word embedding space","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_cosine_similarity(model, \"king\", \"prince\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"It ranges from -1 to 1, where 1 indicates high similarity, 0 indicates no similarity and -1 indicates opposite directions.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_similar_words(): Find the n most similar words to a given word or embedding vector and return the matching strings","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_similar_words(model, \"king\", 5)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_vector_operation(): Computes 1 of 4 vector calculations on two input words or vectors depending on the input operator","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_vector_operation(model, \"king\", \"queen\",\"+\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"or","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> get_vector_operation(model, king_vec, \"queen\",\"euclid\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"get_word_analogy(): Performs word analogy calculations (e.g. king - queen + woman = man)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> word_analogy(model, \"king\", \"queen\", [\"woman\"])","category":"page"},{"location":"getting_started/#Display-Data","page":"Getting Started","title":"Display Data","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"showrelations(): Creates a [PCA Projection](https://en.wikipedia.org/wiki/Principalcomponent_analysis) to 2D of words with connecting vectors ","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> show_relations(\"berlin\", \"germany\", \"paris\", \"france\", \"rome\", \"apple\", wv=model, save_path=\"my_custom_plot.png\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"<div align=\"center\">   <img src=\"assets/PCAProjection.png\" alt=\"Logo\" width=\"400\" height=\"250\" /> </div>","category":"page"},{"location":"getting_started/#3)-Running-a-large-example","page":"Getting Started","title":"3) Running a large example","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"As an alternative (larger) example use a text corpus from e.g. FastText (.bin & .vec file) with about 33 million words. Store this file in the current working directory and apply the same functions as in the previous example.","category":"page"},{"location":"getting_started/#For-Developers","page":"Getting Started","title":"For Developers","text":"","category":"section"},{"location":"getting_started/#1)-Download-the-code","page":"Getting Started","title":"1) Download the code","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"git clone https://github.com/graupnej/GroupIWord2Vec.jl.git","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Navigate to the cloned directory and launch julia. Activate the project environment to tell Julia to use the Project.toml","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> using Pkg\njulia> Pkg.activate(\".\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Resolve dependencies and create a Manifest.toml file","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.instantiate()","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"Precompile the project to ensure all dependencies and the code is ready","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.precompile()","category":"page"},{"location":"getting_started/#2)-Run-tests","page":"Getting Started","title":"2) Run tests","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"To verify everything is working correctly run the code coverage tests","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.test(\"GroupIWord2Vec\")","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"This covers all the tests. To execute a specific test (e.g. Functions)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"julia> Pkg.test(\"GroupIWord2Vec\", test_args=[\"Functions\"])","category":"page"},{"location":"getting_started/#Dependencies","page":"Getting Started","title":"Dependencies","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The package relies on the following non-standard Julia packages:","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"   DelimitedFiles        # Provides functionality for reading and writing delimited text files\n   LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra\n   Plots                 # For visualization functions\n   Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)\n   Statistics            # For basic statistical operations (mean, std, var, etc.)","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The files Project.toml and Manifest.toml in the created environment manage dependencies.","category":"page"},{"location":"getting_started/#References","page":"Getting Started","title":"References","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The text corpus for the simple example (text8) is a preprocessed version of the first 100 million bytes of the English Wikipedia dump from March 3, 2006. It has been filtered to include only lowercase letters (a–z) and spaces, reducing the dataset's size to approximately 100 MB. It is commonly used for training and evaluating language models.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"The text corpus for the large example were obtained using the skip-gram model described in Bojanowski et al. (2016) with default parameters.","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"   P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, Enriching Word Vectors with Subword Information","category":"page"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"As inspiration on how to properly develop julia packages, organize source files, write meaningful tests and more read here.","category":"page"},{"location":"getting_started/#Contributors","page":"Getting Started","title":"Contributors","text":"","category":"section"},{"location":"getting_started/","page":"Getting Started","title":"Getting Started","text":"(Image: Contributors)","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = GroupIWord2Vec","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Coverage) (Image: Build Status) (Image: Dev)","category":"page"},{"location":"#Explanation","page":"Home","title":"Explanation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Word Embeddings are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks.","category":"page"},{"location":"#Prerequisities-and-dependencies","page":"Home","title":"Prerequisities and dependencies","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package uses (Image: Julia) and relies on the following non-standard Julia packages:","category":"page"},{"location":"","page":"Home","title":"Home","text":"   DelimitedFiles        # Provides functionality for reading and writing delimited text files\n   LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra\n   Plots                 # For visualization functions\n   Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)\n   Statistics            # For basic statistical operations (mean, std, var, etc.)","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"}]
}
