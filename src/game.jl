## Difference between Game and GameState is that Game holds current GameState and tracks currentPlayer
## unlike GameState which only keeps currentPlayer at the moment when State was created
abstract type AbstractGame end

struct Connect4 <: AbstractGame
  state::GameState
  player::Int
  value::Int
  done::Bool
end

next_player(game::Connect4) = game.player == 1 ? 2 : 1
name(game::Connect4) = "connect4"
action_size(game::Connect4) = length(action_space(game))

function step(game::Connect4, action)
  next_state, value, done = takeAction(game.state, action)

  return Connect4(next_state, next_player(game), value, done)
end

function take_action(game::Connect4, action)
end
