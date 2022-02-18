using Random
include("../lib/banana.jl")

function gen_xor_data()
    x = bitrand(2)
    y = xor.(x...)
    return x, y
end

# TODO:
# modularize model/prediction to be able to pass various architectures
# implement gradient descent using flux, find appropriate loss function
# surely have to create some container for both binary weights and float weights (used for back prop)

function main()
    ws = [init_weights(2, [2, 3, 1]) for _ in 1:100]
    xor_model_weights::Vector{BitArray}=[] 
    epoch = 1
    while true
        x, y = gen_xor_data()
        γs = predict(x, ws)
        if [y] in γs
            println("found right combination! epoch: $epoch")
            w = ws[findfirst(isequal(y), map(x -> x[1], γs))]
            println("weights found: ", w)
            xor_model_weights = w
            break
        end
        ws = [init_weights(2, [2, 3, 1]) for _ in 1:100]
        epoch += 1
    end

    predictions::Vector{Bool}= []
    ground_truth::Vector{Bool}= []
    for _ in 1:10
        x, y = gen_xor_data()
        γ = model(x, xor_model_weights)[1]
        push!(ground_truth, y)
        push!(predictions, γ)
    end

    println("acc: $(accuracy(predictions, ground_truth))")
end

main()