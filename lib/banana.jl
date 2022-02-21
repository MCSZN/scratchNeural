using Random

FullyConnectedWeights = Vector{BitArray}

# apply bitwise "and" then "xor" bitarray
neuron(input::BitVector, weights::BitVector)::Bool = xor.(.&(input, weights)...)

function init_weights(input_shape::Int, sizes::Vector{Int})::FullyConnectedWeights
    # input layer: shape ==> (n_features, n_neurons)
    # next layers: shape ==> (n_neurons_prev, n_neurons_next)
    @assert input_shape == sizes[1]
    @assert length(sizes) >= 2
    [bitrand(sizes[i], sizes[i+1]) for i in 1:length(sizes)-1]
end

function layer(inputs::BitVector, weights::BitArray)::BitArray
    # vectorized version of neuron, applies inputs to all neurons in the next layer
    # inputs_shape: N
    # weights_shape: N, M
    # output_shape: M
    @assert size(inputs)[1] == size(weights)[1] # make sure N1 == N2
    n_features = size(inputs)[1]
    y = .&(inputs, weights)
    xor.([y[feature, :] for feature in 1:n_features]...) # returns BitArray{M}
end

function model(inputs::BitVector, weights::FullyConnectedWeights)::BitVector
    # sequentially apply layer by layer using the weights
    out = layer(inputs, weights[1])
    if length(weights) == 1 return out end
    for weight in weights[2:end]
        out = layer(out, weight)
    end
    out
end

function predict(inputs::BitVector, weights::Vector{FullyConnectedWeights})::Vector{BitVector}
    predictions = Array{BitVector}(undef, length(weights))
    Threads.@threads for index in 1:length(weights)
        predictions[index] = model(inputs, weights[index])
    end
    return predictions
end

function accuracy(predictions::Vector{Bool}, ground_truth::Vector{Bool})::Float64
    tp = length([i == j for (i, j) in zip(predictions, ground_truth) if i == true])
    tn = length([i == j for (i, j) in zip(predictions, ground_truth) if i == false])
    return (tp + tn) / length(predictions)
end

function loss() end