using Plots

import StatsBase: predict
# We want to learn a simple linear regression

f(x) = π * x

x = linspace(0, 1)
plot(x, f(x), legend = false)

# Since we do not know the true function apart from the fact that it is probably
# linear, we need to start with some samples and a guess.

x_sample = rand(17)
plot!(x, f(x), seriestype = :scatter)

# We need to learn the line from the points. There is a common pattern to tackle
# such scenarios.

# We define a mutable type for the model; It holds the parameters of the model
# learning which completes the model

mutable struct LinearModel
    w::Float64 # weight attached to the predictor
end

# Define its constructor; does not need one technically since automagically
# one constructor is free.
function LinearModel()
    return LinearModel(randn())
end

m = LinearModel()
typeof(m)
# Make a predict method using the model

function predict(m::LinearModel, x::T) where T <: Real
    return m.w * x
end

# Now we are ready to predict using this random initial model. The model upon
# initialization, starts off with a random normal draw as the slope of the fit
# passing through the origin

x = linspace(0, 1)
plot(x, f(x), label = "Truth")
plot!(x, predict.(m, x), label = "Predicted")

# It is bad, but how bad?
# Let us define a root mean square error (RMSE)
function mse(m::LinearModel, x::Vector{T}, y::Vector{T}) where T <: Real
    predicted = predict.(m, x)
    return mean((y .- predicted))^2
end

mse(m, collect(x), collect(f(x)))

# The feature space is fixed at x ∈ (0, 1). We need to choose the value of
# the weight such that the loss is minimized in this space. At minimum loss,
# the gradient of the loss function is 0.

# Hand computing the gradient of the MSE and plugging the formula into a
# function: ∇mse = 2x(wx - y)

function gradient_mse(m::LinearModel, x::T, y::T) where T <: Real
    return 2 * x * (m.w * x - y)
end

function gradient_mse(m::LinearModel, xys::Vector{Tuple{T, T}}) where T <: Real
    return mean(gradient_mse(m, x, y) for (x, y) in xys)
end

# So we need to iterate over a region of the parameter space and figure out the
# point where the gradient is minimized
mu = 0.1 # Step size, called the learning rate
m = LinearModel()

models = LinearModel[deepcopy(m)] # create an array of linear models
xys = [(x, f(x)) for x in rand(17)] # sample data

# Gradient descent

for _ in 1:128
    m.w -= mu * gradient_mse(m, xys)
    push!(models, deepcopy(m))
end

plot(xys, label = "Truth")
for m in models
    plot!([(x, predict(m, x)) for (x, y) in xys])
end
