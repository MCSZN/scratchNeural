using Random

# apply bitwise "and" then "xor" bitarray
neuron(input::BitVector, weights::BitVector)::Bool = xor.(.&(input, weights)...)

function layer(inputs, weights)
    # vectorized version of neuron
    @assert size(inputs)[1] == size(weights)[1]
    n_features = size(inputs)[1]
    y = .&(inputs, weights)
    xor.([y[feature, :] for feature in 1:n_features]...)
end

function init_weights(input_shape::Int, sizes::Vector{Int})::Vector{BitArray}
    @assert input_shape == sizes[1]
    @assert length(sizes) >= 2
    [bitrand(sizes[i], sizes[i+1]) for i in 1:length(sizes)-1]
end

# input layer: shape ==> (n_features, n_neurons)
# next layers: shape ==> (n_neurons_prev, n_neurons_next)

function model(x::BitVector, weights::Vector{BitArray})::BitVector
    out = layer(x, weights[1])
    if length(weights) == 1 return out end
    for weight in weights[2:end]
        out = layer(out, weight)
    end
    out
end