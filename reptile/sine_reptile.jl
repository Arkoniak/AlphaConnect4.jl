# Based on https://blog.openai.com/reptile/
# PyTorch gist implemented in Flux

using Flux
using Knet

srand(0)

runif(low::Float64, high::Float64) = (high - low)*rand() + low
pytorch_init(dims...) = 1/sqrt(dims[1])*rand(dims...)

struct SineTask
    phase::Float64
    ampl::Float64
    x::Array{Float64, 2}
    y::Array{Float64, 2}
end
function SineTask(; min_phase = 0.0, max_phase = 2π,
    min_ampl = 0.1, max_ampl = 5.0,
    xmin = -5.0, xmax = 5.0, npoints = 50)

    phase = runif(min_phase, max_phase)
    ampl = runif(min_ampl, max_ampl)
    x = collect(linspace(xmin, xmax, npoints))
    y = ampl*sin.(x + phase)

    SineTask(phase, ampl, reshape(x, 1, :), reshape(y, 1, :))
end

function SineTask(phase::Float64, ampl::Float64; xmin = -5.0, xmax = 5.0, npoints = 50)
    x = collect(linspace(xmin, xmax, npoints))
    y = ampl*sin.(x + phase)
    SineTask(phase, ampl, reshape(x, 1, :), reshape(y, 1, :))
end

abstract type AbstractReptileModel end
struct FluxReptile{T} <: AbstractReptileModel
    model::T
end
# function ReptileModel()
#     ReptileModel(Chain(Dense(1, 64, tanh; initW = pytorch_init, initb = pytorch_init),
#                        Dense(64, 64, tanh; initW = pytorch_init, initb = pytorch_init),
#                        Dense(64, 1; initW = pytorch_init, initb = pytorch_init)))
# end

function FluxReptile()
    FluxReptile(Chain(Dense(1, 64, tanh),
                      Dense(64, 64, tanh),
                      Dense(64, 1)))
end

predict(m::FluxReptile, x) = m.model(x)

loss(m::RM, x, y) where {RM <: AbstractReptileModel} =
    mean(abs2, predict(m, x) .- y)

function train_on_batch!(m::RM, batch_x, batch_y, innerstepsize) where {RM <: AbstractReptileModel}
    l = loss(m, batch_x, batch_y)
    Flux.Tracker.back!(l)
    for p in params(m.model)
        p.data .-= innerstepsize * p.grad
        p.grad .= 0.0
    end
end

train_on_batch!(m::RM, task::SineTask, ids::Vector{Int}, innerstepsize) where {RM <: AbstractReptileModel} =
  train_on_batch!(m, task.x[:, ids], task.y[:, ids], innerstepsize)

function train!(m::RM; tasks = SineTask[], eval_task = SineTask(), inner_epochs = 1, ntrain = 10,
               outerstepsize0 = 0.1, niter = 30_000,
               innerstepsize = 0.02, ninneriter = 32) where {RM <: AbstractReptileModel}

    eval_ids = randperm(length(eval_task.x))[1:ntrain]
    f_output = open("reptile.csv", "w")
    res = Vector{Float64}()
    res2 = Vector{Float64}()
    !isempty(tasks) && (niter = length(tasks))
    for iter in 1:niter
        weights_before = deepcopy([x.data for x in params(m.model)])
        # Do SGD on task
        if !isempty(tasks)
            f = tasks[iter]
        else
            f = SineTask()
        end
        inds = randperm(length(f.x))
        for _ in 1:inner_epochs
            for start in 1:ntrain:length(f.x)
                mbinds = inds[start:(start + ntrain - 1)]
                train_on_batch!(m, f, mbinds, innerstepsize)
            end
        end

        # Interpolate between current weights and trained weights from this task
        # I.e. (weights_before - weights_after) is the meta-gradient
        outerstepsize = outerstepsize0 * (1.0 - iter / niter) # linear schedule

        for (d1, d2) in zip(weights_before, params(m.model))
            d2.data .= d1 .+ outerstepsize*(d2.data .- d1)
        end

        model_before = deepcopy(m)
        for inneriter in 1:ninneriter
            train_on_batch!(m, eval_task, eval_ids, innerstepsize)
        end
        push!(res, Flux.data(loss(m, eval_task.x, eval_task.y)))
        m = model_before

        if (iter % 1000 == 0)
            # weights_before = deepcopy([x.data for x in params(m.model)])
            model_before = deepcopy(m)
            for inneriter in 1:ninneriter
                train_on_batch!(m, eval_task, eval_ids, innerstepsize)
            end
            println("Iteration $iter: Loss = $(loss(m, eval_task.x, eval_task.y))")
            m = model_before
            # for (d1, d2) in zip(weights_before, params(m.model))
            #     d2.data .= d1
            # end
        end
    end

    for x in res
        write(f_output, "$x\n")
    end
    close(f_output)
end

tasks = [SineTask() for _ in 1:30_000]
model = ReptileModel()
train!(model, tasks = tasks)

f1 = SineTask()
train_ids = randperm(50)[1:10]
for iter in 1:32
    (iter % 8 == 0) && (println("Iteration $iter, loss = $(loss(model, f1.x, f1.y))"))
    train_on_batch!(model, f1, train_ids, 0.02)
end

########## misc
# m = ReptileModel2(64)
m = ReptileModel()
f1 = SineTask()

for iter in 1:10_000
    (iter % 200 == 0) && (println("Iteration $iter, loss = $(loss(m, f1.x, f1.y))"))
    train_on_batch!(m, f1.x, f1.y, 0.02)
end

length(f1.x)
collect(1:10:length(f1.x))

l = loss(m, f1.x, f1.y)
@show m.model[1].W.grad

@show f1.x
@show f1.y
@show m.model(f1.x).data
@show loss(m, f1.x, f1.y)
train_on_batch!(m, f1.x, f1.y, 0.02)

function ReptileModel2(n = 64)
    ReptileModel(Chain(Dense(1, n, tanh), Dense(n, 1)))
end

@show model.model(reshape(f1.x, 1, :)).data
@show reshape(f1.y, 1, :)
loss(model, reshape(f1.x, 1, :), reshape(f1.y, 1, :))
f1.x[:, 1:10]
123 %% 100
f1.x[:, 1:10]

##################
# Cross Check with PyTorch implementation
eval_task = SineTask(3.448296944257913, 3.604427895224856)
eval_ids = [3, 3, 39, 9, 19, 21, 36, 23, 6, 24] + 1
outerstepsize0 = 0.1

m = ReptileModel(Chain(Dense(1, 2, tanh), Dense(2, 1)))
t1 = SineTask(6.054871697856187, 1.978863442246311)
@show t1.x, t1.y
@show m.model(t1.x).data
m.model[1].W.data .= reshape([-0.00748682, 0.53644359], 2, :)
m.model[1].b.data .= [-0.82304513, -0.73593903]
m.model[2].W.data .= reshape([-0.27234524, 0.18961591], 1, :)
m.model[2].b.data .= [-0.01401001]

inds = [27 12 35 33 31 49 22 42 47 26 40  6 41 36  4 43 21  2 11  3 10 32 48  1 34 7 14 28 30 29 19 44 45 18  0 15  5 16 20  9  8 13 25 37 17 24 46 23 39 38]
inds = reshape(inds, 50) + 1

weights_before = deepcopy([x.data for x in params(m.model)])

for _ in 1:1
    for start in 1:10:length(t1.x)
        mbinds = inds[start:(start + 10 - 1)]
        train_on_batch!(m, t1, mbinds, 0.02)
    end
end

@show m.model[1].W.data
@show m.model[1].b.data
@show m.model[2].W.data
@show m.model[2].b.data

outerstepsize = outerstepsize0 * (1.0 - 0 / 3) # linear schedule

for (d1, d2) in zip(weights_before, params(m.model))
    d2.data .= d1 .+ outerstepsize*(d2.data .- d1)
end

model_before = deepcopy(m)
for inneriter in 1:32
    train_on_batch!(m, eval_task, eval_ids, 0.02)
end
# loss(m, eval_task.x, eval_task.y)
println("Loss = $(loss(m, eval_task.x, eval_task.y))")
m = model_before

@show m.model[1].W.data
@show m.model[1].b.data
@show m.model[2].W.data
@show m.model[2].b.data
train_on_batch!(m, eval_task, eval_ids, 0.02)


@show eval_task.x[eval_ids]
@show eval_task.y
@show m.model(eval_task.x).data

t2 = SineTask(4.3833460800803135, 0.3951048109834222)
inds = [48 36 25  8 19 22  7 10 45 28 33 41  2  1 44 32 30 43 29  9  5 17 16 24 21 13 31 23 26 34 39 15 42 40  3 37  6  4 47 20 12 49 14  0 27 18 46 11 38 35]
inds = reshape(inds, 50) + 1

weights_before = deepcopy([x.data for x in params(m.model)])

for _ in 1:1
    for start in 1:10:length(t2.x)
        mbinds = inds[start:(start + 10 - 1)]
        train_on_batch!(m, t2, mbinds, 0.02)
    end
end

@show m.model[1].W.data
@show m.model[1].b.data
@show m.model[2].W.data
@show m.model[2].b.data

outerstepsize = outerstepsize0 * (1.0 - 1 / 3) # linear schedule

for (d1, d2) in zip(weights_before, params(m.model))
    d2.data .= d1 .+ outerstepsize*(d2.data .- d1)
end

model_before = deepcopy(m)
for inneriter in 1:32
    train_on_batch!(m, eval_task, eval_ids, 0.02)
end
# loss(m, eval_task.x, eval_task.y)
println("Loss = $(loss(m, eval_task.x, eval_task.y))")
m = model_before

#########
# Linear regression....

a = 1.0
b = 2.0

layer = Dense(1, 1)
layer.W.data .= reshape([0.804456], (1, 1))
layer.b.data .= [0.173496]
x = reshape(collect(linspace(0, 10, 10)), 1, :)
y = a * x + b
 # .+ reshape(randn(1000), 1, :)
loss1(m, x, y) = mean((m(x) .- y).^2)
@show loss1(layer, x, y)

for i in 1:10000
    l = loss1(layer, x, y)
    (i % 1000 == 1) && @show l
    Flux.Tracker.back!(l)

    layer.W.data .-= 0.01(layer.W.grad)
    layer.b.data .-= 0.01(layer.b.grad)

    @. layer.W.grad = 0
    @. layer.b.grad = 0
end

@show layer.W.data
@show layer.b.data

inds = [1, 2, 3, 4, 5, 6, 7]
inds[1:3]

collect(1:10:50)

##################
m1 = Chain(Dense(1, 64, tanh), Dense(64, 64, tanh), Dense(64, 1))
loss(m, x, y) = mean(abs2, m(reshape(x, 1, :)) .- reshape(y, 1, :))
loss(m1, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])

l = loss(m1, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])
Flux.Tracker.back!(l)
for p in params(m1)
    p.data .-= 0.02 * p.grad
    p.grad .= 0.0
end

w1 = deepcopy(Flux.data.(params(m1)))

function predict(w, x)
  x = tanh.(w[1] * reshape(x, 1, :) .+ w[2])
  x = tanh.(w[3]*x .+ w[4])
  x = w[5]*x .+ w[6]
end

loss2(w, x, y) = mean(abs2, predict(w, x) .- reshape(y, 1, :))
loss2(w1, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])
lossgrad = grad(loss2)

l = lossgrad(w1, [0.1, 0.2, 0.3], [10.0, 11.0, 12.0])
for i in 1:length(l)
    w1[i] -= 0.02 * l[i]
end


predict(w1, [0.1, 0.2, 0.3])
