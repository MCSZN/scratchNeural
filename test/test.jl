using Random
include("../lib/banana.jl")

function gen_xor_data()
    x = bitrand(2)
    y = xor.(x...)
    return x, y
end

# TODO: move scoring to banana lib
# modularize model/prediction to be able to pass various architectures

function main()
    ws = [init_weights(2, [2, 3, 1]) for _ in 1:10]
    xor_model_weights::Vector{BitArray}=[] 
    epoch = 1
    while true
        x, y = gen_xor_data()
        γs = predict(x, ws)
        println(γs)
        if [y] in γs
            println("found right combination! epoch: $epoch")
            w = ws[findfirst(isequal(y), map(x -> x[1], γs))]
            println(w)
            xor_model_weights = w
            break
        end
        ws = [init_weights(2, [2, 3, 1]) for _ in 1:10]
        epoch += 1
    end

    predictions = []
    ground_truth = []
    for _ in 1:10
        x, y = gen_xor_data()
        γ = model(x, xor_model_weights)[1]
        push!(ground_truth, y)
        push!(predictions, γ)
    end

    tp = length([i == j for (i, j) in zip(predictions, ground_truth) if i == true])
    tn = length([i == j for (i, j) in zip(predictions, ground_truth) if i == false])

    println("mean accuracy = $((tp + tn) / length(predictions))")
end

main()