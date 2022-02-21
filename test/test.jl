using Random
include("../lib/banana.jl")

function gen_xor_data()
    x = bitrand(2)
    y = xor.(x...)
    return x, y
end

# TODO:
# first:
#   find better test than xor, that one is too easy => find appropriate dataset
#   implement gradient descent using flux, find appropriate loss function
# second: surely have to create some container for both binary weights and float weights (used for back prop)
# last: modularize model/prediction to be able to pass various architectures
function score_model(weights::FullyConnectedWeights)::Float64
    predictions::Vector{Bool}= []
    ground_truth::Vector{Bool}= []
    for _ in 1:10
        x, y = gen_xor_data()
        γ = model(x, weights)[1]
        push!(ground_truth, y)
        push!(predictions, γ)
    end
    score = accuracy(predictions, ground_truth)
    println("acc: $score")
    return score
end

function main()
    ws = [init_weights(2, [2, 3, 1]) for _ in 1:2]
    xor_model_weights::FullyConnectedWeights=[]
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

    score_model(xor_model_weights)
end

main()