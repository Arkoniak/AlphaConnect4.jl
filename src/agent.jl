abstract type AbstractAgent end

struct Agent <: AbstractAgent
  root::Node
  cpuct::???
  mcts::MCTS
  model::???
end

function replay!(agent::AbstractAgent, memory::Memory)
end

function simulate(agent::AbstractAgent)
end

function act(agent::AbstractAgent, ???)
end

function get_preds(agent, state)
end

function evaluate_leaf(agent::AbstractAgent, leaf, value, done, breadcrumbs)
  if done == 0
    value, probs, allowed_actions = agent.get_preds(leaf.state)
    probs = probs[allowed_actions]

    for (idx, action) in enumerate(allowed_actions)
      new_state, _, _ = take_action(leaf.state, action)
      if new_state.id ∈ agent.mcts.nodes
        node = agent.mcts.nodes[new_state.id]
      else
        node = Node(new_state)
        add_node(agent.mcts, node)
      end

      new_edge = Edge(leaf, node, probs[idx], action)
      add_edge(agent.mcts, leaf, new_edge)
      #= push!(leaf.edges, new_edge) =#
    end
  else
    ## Log game value
  end

  return (value, breadcrumbs)
end

function get_av(agent::AbstractAgent, tau)
  edges = agent.mcts.root.edges
  ppi = zeros(Int, agent.action_size)
  values = zeros(Float64, agent.action_size)

  for edge in edges
    ppi[edge.action] = edge.stats[:N]^(one(tau)/tau)
    values[edge.action] = edge.stats[:Q]
  end

  ppi ./= sum(ppi)

  return ppi, values
end

function choose_action(agent::AbstractAgent, ppi, values, tau)
  ## Note: tau is unused?
  if tau == 0
    action = rand(find(ppi .== max(ppi)))
  else
    action = indmax(rand(Multinomial(1, ppi)))
  end

  value = values[action]

  return action, value
end

predict(agent::AbstractAgent, input_to_model) = predict(agent.model, input_to_model)

function buildMCTS(agent::AbstractAgent, state)
  logg("building new mcts tree")
  agent.root = Node(state)
  agent.mcts(agent.root, agent.cpuct)
end

change_rootMCTS(agent::AbstractAgent, state) = (agent.mcts.root = agent.mcts.nodes[state.id])

###################

struct PlayResult
end

function play_match(agent1::AbstractAgent, agent2::AbstractAgent)
end

function play_matches(player1::AbstractAgent, player2::AbstractAgent)
end
