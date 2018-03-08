# Based on https://blog.openai.com/reptile/
# PyTorch gist implemented in Flux

using Flux

runif(low::Float64, high::Float64) = (high - low)*rand() + low

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

f_randomsine(t::SineTask, x::Float64) = t.ampl*sin(x + t.phase)
f_randomsine(t::SineTask, x::Vector{Float64}) = t.ampl*sin.(x + t.phase)

abstract type AbstractReptileModel end
struct ReptileModel{T} <: AbstractReptileModel
    model::T
end
function ReptileModel()
    ReptileModel(Chain(Dense(1, 64, tanh), Dense(64, 64, tanh), Dense(64, 1)))
end

loss(m::RM, x, y) where {RM <: AbstractReptileModel} =
    mean((m.model(x) .- y).^2)

function train_on_batch(m::RM, batch_x, batch_y, innerstepsize) where {RM <: AbstractReptileModel}
    l = loss(m, reshape(batch_x, 1, :), reshape(batch_y, 1, :))
    Flux.Tracker.back!(l)
    for p in params(m.model)
        p.data .-= innerstepsize * p.grad
    end
end

function train!(m::RM; inner_epochs = 1, ntrain = 10,
               outerstepsize0 = 0.1, niter = 30_000,
               innerstepsize = 0.02) where {RM <: AbstractReptileModel}
    for iter in 1:niter
        weights_before = deepcopy([x.data for x in params(m.model)])
        # Do SGD on task
        f = SineTask()
        (iter % 1000 == 0) && println("Iteration $iter: Loss before = $(loss(m, reshape(f.x, 1, :), reshape(f.y, 1, :)))")
        inds = randperm(length(f.x))
        for _ in 1:inner_epochs
            for start in 1:ntrain:length(f.x)
                mbinds = inds[start:(start + ntrain - 1)]
                train_on_batch(m, f.x[mbinds], f.y[mbinds], innerstepsize)
            end
        end
        (iter % 1000 == 0) && println("Iteration $iter: Loss after = $(loss(m, reshape(f.x, 1, :), reshape(f.y, 1, :)))")

        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        outerstepsize = outerstepsize0 * (1.0 - iter / niter) # linear schedule

        for (d1, d2) in zip(weights_before, params(m.model))
            d2.data .= d1 .+ outerstepsize*(d2.data .- d1)
        end
    end
end

model = ReptileModel()
train!(model)

f1 = SineTask()
@show model.model(reshape(f1.x, 1, :)).data
@show reshape(f1.y, 1, :)
loss(model, reshape(f1.x, 1, :), reshape(f1.y, 1, :))

123 %% 100
