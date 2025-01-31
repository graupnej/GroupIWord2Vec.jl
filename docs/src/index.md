```@meta
CurrentModule = GroupIWord2Vec
```
<div align="center">
  <img src="assets/WordEmbeddings.png" alt="Logo" width="250" height="250" />
  <h1>Word2Vec</h1>
  A Julia package that provides tools for running word embedding algorithms.
  <br/>
  <a href="https://julialang.org/downloads/">
    <img src="https://img.shields.io/badge/Julia-v1.10-blue" alt="Julia Version"/>
  </a>
</div>

## Prerequisities and dependencies
This package uses [![Julia](https://img.shields.io/badge/Julia-v1.10-blue)](https://julialang.org/downloads/) and relies on the following non-standard Julia packages:

       DelimitedFiles        # Provides functionality for reading and writing delimited text files
       LinearAlgebra         # Offers a suite of mathematical tools and operations for linear algebra
       Plots                 # For visualization functions
       Word2vec.jll          # Links to the underlying Word2Vec implementation (C code)
       Statistics            # For basic statistical operations (mean, std, var, etc.)

```@index
```
