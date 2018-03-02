## Note: there are two possible implementations of tree structure
## 1. Abstract types + cross references
## 2. Concrete types with integer ids and vectors of edges/nodes
## They should be tested for speed, I'll start with the second one.

using Distributions

########################
# n = number of games
# w = number of wins
# q = probability of winning (w/n)
# p = ???
mutable struct EdgeStat
  n::Int
  w::Int
  p::Float64
  q::Float64
end

struct Edge
  out_node::Int
  stats::EdgeStat
end


########################
struct Node
  state
  player_turn
  id
  edges::Vector{Edge}
end
function Node(state)
  Node(state, state.player_turn, state.id, Vector{Int}())
end

is_leaf(node::Node) = isempty(node.edges)
length(node::Node) = length(node.edges)

########################
mutable struct MCTS
  root::Node
  nodes::Dict{Int, Node}
end

add_node(mcts::MCTS, node::Node) = (mcts.nodes[node.id] = node)
add_edge(mcts::MCTS, node::Node, edge::Edge) = push!(node.edges, edge)
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
      nb += edge.stats.n
    end

    for (idx, edge) in enumerate(current_node.edges)
      u = mcts.cpuct * ((one(epsilon) - epsilon) * edge.stats.p + epsilon * nu[idx]) * sqrt(nb)/(one(epsilon) + edge.stats.n)
      q = edge.stats.q
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
    edge.stats.n += 1
    edge.stats.w += value*direction
    edge.stats.q = edge.stats.w / edge.stats.n
  end
end
