abstract type AbstractAgent end

struct Agent <: AbstractAgent
end

function replay!(agent::AbstractAgent, memory::Memory)
end

function simulate(agent::AbstractAgent)
end

function act(agent::AbstractAgent, ???)
end

function get_preds(agent, state)
end

###################

struct PlayResult
end

function play_match(agent1::AbstractAgent, agent2::AbstractAgent)
end

function play_matches(player1::AbstractAgent, player2::AbstractAgent)
end
