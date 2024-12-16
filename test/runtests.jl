using Test
using GroupIWord2Vec

# Adjust paths to match your package structure
const BASE_DIR = dirname(@__DIR__)  # Go up one level from test/ to get to package root
const TEXT_MODEL_PATH = joinpath(BASE_DIR, "wiki.en", "wiki.en.vec")
const BINARY_MODEL_PATH = joinpath(BASE_DIR, "wiki.en", "wiki.en.bin")

function compare_word_embeddings(text_model_file::String, binary_model_file::String, test_words::Vector{String}, tolerance::Float64=1e-5)
    println("\nLoading text model...")
    text_model = load_text_model(text_model_file)
    println()
    
    println("Loading binary model...")
    binary_model = load_fasttext_embeddings(binary_model_file)
    println()
    
    println("\n=== Comparison Results ===")
    
    # Compare vocabulary sizes
    text_vocab_size = length(text_model.vocab)
    binary_vocab_size = length(binary_model.vocab)
    println("\nVocabulary size comparison:")
    println("Text model: $text_vocab_size words")
    println("Binary model: $binary_vocab_size words")
    if text_vocab_size != binary_vocab_size
        println("⚠️  WARNING: Vocabulary sizes differ by $(abs(text_vocab_size - binary_vocab_size)) words")
    end
    
    # Compare vector dimensions
    text_dim = size(text_model.vectors, 1)
    binary_dim = size(binary_model.vectors, 1)
    println("\nVector dimension comparison:")
    println("Text model: $text_dim dimensions")
    println("Binary model: $binary_dim dimensions")
    if text_dim != binary_dim
        println("⚠️  WARNING: Vector dimensions do not match")
    end
    
    # Compare specific test words and their vectors
    println("\nWord-specific comparisons:")
    for word in test_words
        println("\nChecking word: '$word'")
        
        text_has_word = haskey(text_model.vocab, word)
        binary_has_word = haskey(binary_model.vocab, word)
        
        if !text_has_word
            println("❌ Word not found in text model")
        end
        if !binary_has_word
            println("❌ Word not found in binary model")
        end
        
        if text_has_word && binary_has_word
            text_vector = text_model.vectors[:, text_model.vocab[word]]
            binary_vector = binary_model.vectors[:, binary_model.vocab[word]]
            
            # Compare vectors
            if isapprox(text_vector, binary_vector, rtol=tolerance)
                println("✓ Vectors match within tolerance of $tolerance")
            else
                max_diff = maximum(abs.(text_vector - binary_vector))
                println("⚠️  Vector mismatch detected:")
                println("Maximum difference: $max_diff")
                println("First 5 components comparison:")
                println("Text:   $(text_vector[1:5])")
                println("Binary: $(binary_vector[1:5])")
            end
        end
    end
    println("\n=== End of Comparison ===")
end

@testset "GroupIWord2Vec Tests" begin
    # Test words to compare
    test_words = ["king", "of", "in", "and", "to"]
    
    # Run comparison tests
    compare_word_embeddings(TEXT_MODEL_PATH, BINARY_MODEL_PATH, test_words)
end