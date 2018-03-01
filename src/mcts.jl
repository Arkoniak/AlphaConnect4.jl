## Note: there are two possible implementations of tree structure
## 1. Abstract types + cross references
## 2. Concrete types with integer ids and vectors of edges/nodes
## They should be tested for speed, I'll start with the second one.

using Distributions

struct Node
  state
  player_turn
  id
  edges
end
function Node(state)
  Node(state, state.player_turn, state.id, Vector{Int}())
end

is_leaf(node::Node) = isempty(node.edges)
length(node::Node) = length(node.edges)

########################
struct Edge
  out_node::Node
  stats::Dict{Symbol, Float64}
end

########################
struct MCTS
  root::Node
  nodes::Dict{Int, Node}
  edges::Dict{Int, Edge}
end

add_node(mcts::MCTS, node::Node) = (mcts.nodes[node.id] = node)
length(mcts::MCTS) = length(mcts.nodes)

function move_to_leaf(mcts::MCTS)
  breadcrumbs = Vector{Int}()
  current_node = mcts.root

  done = 0
  value = 0

  while !is_leaf(current_node)
    ## TODO: Remove this hack
    maxQU = -99999

    if current_node == mcts.root
      epsilon = Config.EPSILON
      nu = rand(Dirichlet(length(current_node), Config.ALPHA))
    else
      epsilon = 0
      nu = zeros(Float64, length(current_node))
    end

    nb = 0
    for edge in current_node.edges
      nb += edges[edge_id].stats[:N]
    end

    for (idx, edge_id) in enumerate(current_node.edges)
      edge = edges[edge_id]
      u = mcts.cpuct * ((one(epsilon) - epsilon) * edge.stats[:P] + epsilon * nu[idx]) * sqrt(nb)/(one(epsilon) + edge.stats[:N])
      q = edge.stats[:Q]
      if u + q > maxQU
        maxQU = u + q
        simulation_action = edge.action
        simulation_edge = edge
      end
    end

    ???? = take_action(current_node.state, simulation_action)
    current_node = nodes[simulation_edge.out_node]
    push!(breadcrumbs, simulation_edge)
  end

  return current_edge, value, done, breadcrumbs
end

function back_fill(mcts::MCTS, leaf, value, breadcrumbs)
  current_player = leaf.state.player_turn
  for edge in breadcrumbs
    player_turn = edge.player_turn
    direction = player_turn == current_player ? 1 : -1
    edge.stats[:N] += 1
    edge.stats[:W] += value*direction
    edge.stats[:Q] = edge.stats[:W] / edge.stats[:N]
  end
end
