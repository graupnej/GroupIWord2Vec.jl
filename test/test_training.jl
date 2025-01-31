using Test
using GroupIWord2Vec

@testset "create_vocabulary" begin
    f = open("test/data/test.txt", "w")
    write(f, """The quick brown fox jumps over the lazy dog 
                while the cat naps peacefully under a tree in the sunshine.
                The quick brown rabbit hops over the sleepy turtle while the bird chirps peacefully under a tree in the sunshine.
                A swift grey rabbit leaps past a drowsy turtle as a sparrow sings softly beneath the shade of the tree.""")
    close(f)
    @testset "basic functionality" begin
        voc = create_vocabulary("test/data/test.txt")
        @test isa(voc, Dict{String, Int})   # Check output type
        @test length(voc) > 0             # Ensure vocabulary is not empty
        @test length(voc) == 35           #check if all words are loaded
        @test "the" in keys(voc)          # Check common word is included
        @test "quick" in keys(voc)        # Check common word is included
        @test voc["fox"] != voc["quick"] # Ensure unique indices    
        @test all(k -> k == lowercase(k), keys(voc)) #check lowercase
    end
    
    @testset "edge cases" begin
        f = open("test/data/empty.txt", "w")
        write(f, "")
        close(f)
        voc = create_vocabulary("test/data/empty.txt")
        @test isempty(voc)  
    end
    
end

@testset "sequence_text" begin
    @testset "basic functionality" begin
        voc = create_vocabulary("test/data/test.txt")
        seq = GroupIWord2Vec.sequence_text("test/data/test.txt", voc)
        @test isa(seq, Vector{Int})   # Check output type
        @test length(seq) > 0             # Ensure vocabulary is not empty
        @test length(seq) == 60           #check if all words are loaded
        @test last(seq) == voc["tree"]
        @test all(i -> i <= length(voc), seq) #check lowercase
    end
    
    @testset "edge cases" begin
        f = open("test/data/empty.txt", "w")
        write(f, "")
        close(f)
        voc = create_vocabulary("test/data/empty.txt")
        @test isempty(voc)  
    end
    
end

@testset "create_custom_model" begin
    
    @testset "basic functionality" begin
        model = create_custom_model(10, 20)
        @testset "basic functionality" begin
            #@test isa(model, Flux.Chain)
            @test size(model[1].weight)[1] == 10             
            @test size(model[1].weight)[2] == 20           
            @test isapprox(sum(model([1])), 1)           
        end
    end            
    @testset "edge cases" begin
        @test true  
    end
    
end

@testset "train_custom_model" begin
    
    @testset "basic functionality" begin
        voc = create_vocabulary("test/data/test.txt")
        the_i = voc["the"]
        brown_i = voc["brown"]
        quick_i = voc["quick"]
        model = create_custom_model(10, length(voc))
        model_out_before = model([the_i, brown_i])[quick_i]
        new_model = train_custom_model(model, "test/data/test.txt", voc, 10, 1, batchsize=5)
        model_out_after = new_model([the_i, brown_i])[quick_i]
        @test model_out_before < model_out_after
        #@test isa(new_model, Flux.Chain)   # Check output type
        @test size(new_model[1].weight) == size(model[1].weight)      
        @test isapprox(sum(new_model([1])), 1)           
        
        
    end
    
    @testset "edge cases" begin
        @test true  
    end
    
end

@testset "save_custom_model" begin
    
    @testset "basic functionality" begin
        
        voc = create_vocabulary("test/data/test.txt")
        model = create_custom_model(10, length(voc))
        save_custom_model(model, voc, "test/data/saved.txt")
        @test isfile("test/data/saved.txt")
    end
    
    @testset "edge cases" begin
        @test true  
    end
    
end