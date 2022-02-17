using Random
include("../lib/banana.jl")

function gen_xor_data()
    x = bitrand(2)
    y = xor.(x...)
    return x, y
end


function main()
    w = init_weights(2, [2, 3, 1])
    while true
        x, y = gen_xor_data()
        γ = model(x, w)[1]
        if y == γ
            println("found right combination!")
            println(w)
            break
        end
        w = init_weights(2, [2, 1])
    end
end

main()