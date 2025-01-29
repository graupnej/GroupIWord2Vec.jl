```@meta
CurrentModule = GroupIWord2Vec
```

[![Coverage](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/graupnej/GroupIWord2Vec.jl)
[![Build Status](https://github.com/graupnej/GroupIWord2Vec.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/graupnej/GroupIWord2Vec.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://graupnej.github.io/GroupIWord2Vec.jl/dev/)

## Explanation

[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding) are numerical representations of words in a high-dimensional vector space, where words with similar meanings are positioned closer together. These vectors capture semantic relationships between words, allowing machines to understand language context and meaning through mathematical operations. They serve as the foundation for many natural language processing tasks.

## Prerequisities and dependencies
This package uses [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) and relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra
       Plots                 # For visualization functions
       Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)
       Statistics            # For basic statistical operations (mean, std, var, etc.)

```@index
```
