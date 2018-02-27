struct Game
  board::Array{UInt8, 2}
  player::Int
  value::Int
  done::Bool
end

next_player(game::Game) = game.player == 1 ? 2 : 1

function step(game::Game, action)
  next_state, value, done = takeAction(game.state, action)

  return Game(next_state, next_player(game), value, done)
end
