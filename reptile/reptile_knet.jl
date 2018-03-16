# Based on https://blog.openai.com/reptile/
# PyTorch gist implemented in Flux

using Knet

srand(0)

runif(low::Float64, high::Float64) = (high - low)*rand() + low
pytorch_init(dims...) = 1/sqrt(dims[1])*rand(dims...)

struct SineTask
    phase::Float64
    ampl::Float64
    x::Vector{Float64}
    y::Vector{Float64}
end
function SineTask(; min_phase = 0.0, max_phase = 2π,
    min_ampl = 0.1, max_ampl = 5.0,
    xmin = -5.0, xmax = 5.0, npoints = 50)

    phase = runif(min_phase, max_phase)
    ampl = runif(min_ampl, max_ampl)
    x = collect(linspace(xmin, xmax, npoints))
    y = ampl*sin.(x + phase)

    SineTask(phase, ampl, x, y)
end

function SineTask(phase::Float64, ampl::Float64; xmin = -5.0, xmax = 5.0, npoints = 50)
    x = collect(linspace(xmin, xmax, npoints))
    y = ampl*sin.(x + phase)
    SineTask(phase, ampl, reshape(x, 1, :), reshape(y, 1, :))
end

abstract type AbstractReptileModel end
struct ReptileModel{T} <: AbstractReptileModel
    model::T
end

function predict(w, x)
    x = reshape(x, 1, :)
    # x = tanh.(w[1]*x .+ w[2])
    # x = tanh.(w[3]*x .+ w[4])
    x = w[5]*x .+ w[6]
    reshape(x, :)
end

loss(w, x, y) = mean(abs2, predict(w, x) .- y)
lossgrad = grad(loss)

function train_on_batch!(m::RM, batch_x::Vector{T}, batch_y::Vector{T}, innerstepsize) where {RM <: AbstractReptileModel, T <: AbstractFloat}
    l = lossgrad(m.model, batch_x, batch_y)
    for i in 1:length(l)
        m.model[i] -= innerstepsize * l[i]
    end
end

train_on_batch!(m::RM, task::SineTask, ids::Vector{Int}, innerstepsize) where {RM <: AbstractReptileModel} =
  train_on_batch!(m, task.x[:, ids], task.y[:, ids], innerstepsize)

#############
## Misc
#############
m = ReptileModel([randn(64, 1), zeros(64, 1), randn(64, 64), zeros(64, 1), randn(1, 64), zeros(1, 1)])
predict(m.model, [0.3, 0.2, 0.3])
loss(m.model, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])
lossgrad = grad(loss)
lossgrad(m.model, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])

train_on_batch!(m, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0], 0.02)
print(loss(m.model, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0]))
print(predict(m.model, [0.3, 0.2, 0.3]))


m = function(x)
    x = mat(x)
    w1 = randn(64, 1)
    b1 = zeros(64, 1)
    x = tanh.(w1 * x .+ b1)
    w2 = randn(64, 64)
    b2 = zeros(64, 1)
    x = tanh.(w2 * x .+ b)
end

randn(64, 1) * reshape([0.1, 0.2, 0.3], 1, :) .+ ones(64, 1)
