# Julia_ML_Group_I - Learning Word-Embeddings with Word2Vec

Create a new Julia package that implements some tools for running and testing word-embedding algorithms. The package should include functions for:
  - Training a standard Word2Vec model on some datasets.
  - Saving the trained model.
  - Loading existing pre-trained models.
  - Bonus: Visualize the embeddings with dimension reduction methods like t-SNE.
  - Bonus: Implement ConEc [2] and benchmark the performance.

# Related Julia packages and Python codes:
  - https://github.com/JuliaText/Embeddings.jl
  - https://github.com/JuliaText/Word2Vec.jl
  - https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
  - https://github.com/cod3licious/conec

# Requirements for milestone 1
  - Basic project structure available on GitHub.
  - Some reviewable basic functionality implemented, e.g. functions for loading a pre-trained Word2Vec model from a file in binary and text format and generating word embeddings for a given word using the loaded model.
  - Select a pre-trained Word2Vec model from a reputable source (e.g., Gensim, FastText). You can find various models online.
  - Minimal documentation, including a basic "Getting Started”.

# References
  - Mikolov, Tomas, et al. https://papers.nips.cc/paper/5021-distributedrepresentations-of-words-and-phrases-and-their-compositionality.pdf
  - Horn, Franziska, “Context encoders as a simple but powerful extension of word2vec“ https://arxiv.org/abs/1706.02496. ACL (2017).
  - Joulin, Armand et al.,"FastText.zip: Compressing text classification models", CoRR abs/1612.03651 (2016). https://fasttext.cc
