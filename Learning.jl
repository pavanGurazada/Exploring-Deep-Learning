using Distributions
using Plots

srand(20130810)

"""
Problem
-------
*The Hoeffding experiment*

Flip 1000 fair coins. Each coin is flipped 10 times.

From the distribution of the heads from the results choose 3 coins - the first
one, a randomly chosen coin, and the coin with minimum frequency of heads.

Let nu_1, nu_rand, nu_min, be the fraction of heads obtained for these three
coins. Repeat the experimen 10^5 times and figure out the distribution of the
nu's

"""

function hoeffding_experiment(n_coins::Int, n_tosses::Int, n_repeats::Int)

    result = Matrix{Int}(n_repeats, 3)

    for r in 1:n_repeats
        coin_flips = [rand(Binomial(n_tosses, 0.5)) for coin in 1:n_coins]

        nu_1 = coin_flips[1]
        nu_rand = sample(coin_flips)
        nu_min = minimum(coin_flips)

        result[r, 1] = nu_1
        result[r, 2] = nu_rand
        result[r, 3] = nu_min

    end

    return result
end

@time result = hoeffding_experiment(1000, 10, 10^5)
