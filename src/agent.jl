abstract type AbstractAgent end

struct Agent{M, T} <: AbstractAgent
  name::String
  stat_size::Int
  action_size::Int
  mcts_simulations::Int
  cpuct::T
  model::M

  train_overall_loss::Vector{Float64}
  train_value_loss::Vector{Float64}
  train_policy_loss::Vector{Float64}

  train_overall_loss::Vector{Float64}
  train_value_loss::Vector{Float64}
  train_policy_loss::Vector{Float64}

  val_overall_loss::Vector{Float64}
  val_value_loss::Vector{Float64}
  val_policy_loss::Vector{Float64}

  root::Node
  mcts::MCTS
end

function Agent(name, stat_size, action_size, mcts_simulations, cpuct::T, model::M) where {M, T}
  Agent(name, stat_size, action_size, mcts_simulations, cpuct, model,
        Vector{T}(), Vector{T}(), Vector{T}(),
        Vector{T}(), Vector{T}(), Vector{T}()
       )
end

function replay!(agent::AbstractAgent, memory::Memory)
  for i in 1:Config.TRAINING_LOOPS
    minibatch = get_minibatch(memory)

    training_states = [convert_to_model_input(row["state"]) for row in minibatch]
    training_targets = Dict(
      :value_head => [row["value"] for row in minibatch],
      :policy_head => [row["av"] for row in minibatch]
    )
    fit = fit!(agent.model, training_states, training_targets, 
               epochs=config.EPOCHS, verbose=1, validation_split=0, batch_size = 32)
    push!(agent.train_overall_loss(round(fit.history["loss"][config.EPOCHS],4)))
    push!(agent.train_value_loss(round(fit.history["value_head_loss"][config.EPOCHS],4)))
    push!(agent.train_policy_loss(round(fit.history["policy_head_loss"][config.EPOCHS],4)))
  end

  # TODO: May be draw some plots

  print_weight_averages(agent.model)
end

function simulate(agent::AbstractAgent)
  ##TODO: Lots of logging...

  ##### MOVE THE LEAF NODE
  leaf, value, done, breadcrumbs = move_to_leaf(agent.mcts)

  ##### EVALUATE THE LEAD NODE
  value, breadcrumbs = evaluate_leaf(agent, leaf, value, done, breadcrumbs)

  ##### BACKFILL THE VALUE THROUGH THE TREE
  back_fill(agent.mcts, value, breadcrumbs)
end

function act(agent::AbstractAgent, state, tau)
  # Egg and chicken problem
  if agent.mcts undef || !(state.id ∈ agent.mcts.nodes)
    buildMCTS(agent, state)
  else
    change_rootMCTS(agent, state)
  end

  ##### run the simulation
  for sim in 1:agent.mcts_simulations
    simulate(agent)
  end

  ##### get action values
  ppi, values = get_av(agent, 1.0)

  ##### pick the action
  action, value = choose_action(agent, ppi, values, tau)

  next_state, _, _ = take_action(state, action)

  nn_value = -get_preds(agent, next_state)[1]

  return (action, ppi, value, nn_value)
end

function get_preds(agent, state)
  input_to_model = convert_to_model_input(agent.model, state)
  preds = predict(agent, input_to_model)

  value_array = preds[1]
  logits_array = preds[2]

  value = value_array[1]
  logits = logits_array[1]

  allowed_actions = state.allowed_actions
  mask = fill(true, size(logits))
  mask[allowed_actions] = false
  logits[mask] = -100.0

  # SOFTMAX
  odds = exp.(logits)
  probs = odds ./ sum(odds)

  return (value, probs, allowed_actions)
end

function evaluate_leaf(agent::AbstractAgent, leaf, value, done, breadcrumbs)
  if !done
    value, probs, allowed_actions = get_preds(agent, leaf.state)
    probs = probs[allowed_actions]

    for (idx, action) in enumerate(allowed_actions)
      new_state, _, _ = take_action(leaf.state, action)
      if new_state.id ∈ keys(agent.mcts.nodes)
        node = agent.mcts.nodes[new_state.id]
      else
        node = Node(new_state)
        add_node(agent.mcts, node)
      end

      new_edge = Edge(leaf, node, probs[idx], action)
      add_edge(agent.mcts, leaf, new_edge)
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
    ppi[edge.action] = edge.stats.n^(one(tau)/tau)
    values[edge.action] = edge.stats.q
  end

  ppi ./= sum(ppi)

  return ppi, values
end

function choose_action(agent::AbstractAgent, ppi, values, tau)
  ## Note: is tau just a flag here?
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
