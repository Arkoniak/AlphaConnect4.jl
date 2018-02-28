struct Node
  state
  player_turn
  id
  edges
end
function Node(state)
  Node(state, state.player_turn, state.id, Vector())
end

is_leaf(node::Node) = isempty(node.edges)

########################
struct Edge
end

########################
struct MCTS
end


