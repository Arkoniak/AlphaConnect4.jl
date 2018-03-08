# Based on https://blog.openai.com/reptile/
# PyTorch gist implemented in Flux

using Flux

runif(low::Float64, high::Float64) = (high - low)*rand() + low

struct SineTask
    phase::Float64
    ampl::Float64
end
SineTask(; min_phase = 0.0, max_phase = 2π, min_ampl = 0.1, max_ampl = 5.0) =
    SineTask(runif(min_phase, max_phase), runif(min_ampl, max_ampl))

f_randomsine(t::SineTask, x::Float64) = t.ampl*sin(x + t.phase)
f_randomsine(t::SineTask, x::Vector{Float64}) = t.ampl*sin.(x + t.phase)
f_plot = SineTask()

function train(niter)
    for i in 1:niter

    end
end
