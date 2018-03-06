## Difference between Game and GameState is that Game holds current GameState and tracks currentPlayer
## unlike GameState which only keeps currentPlayer at the moment when State was created
abstract type AbstractGame end
abstract type AbstractGameState end

struct Connect4 <: AbstractGame
  board::Array{UInt8, 2}
  player::Int
  value::Int
  done::Bool
end
Connect4() = Connect4(Connect4State(), 1, 0, false)

next_player(game::Connect4) = game.player == 1 ? 2 : 1
name(game::Connect4) = "connect4"
action_size(game::Connect4) = length(action_space(game))
grid_shape(game::Connect4) = (6, 7)

step(game::Connect4, action) = take_action(game, action)

isinbounds(game::Connect4, pos) =
  pos[2] >= 0 && pos[1] >= 0 && pos[2] <= grid_shape(game, 2) && pos[1] <= grid_shape(game, 1)

function count_indir(game::Connect4, pos, dir)
  cnt = 0
  cur = pos .+ dir
  while isinbounds(game, cur) && game(cur) == game.player
    cnt += 1
    cur .+= dir
  end

  return cnt
end

function is_endgame(game::Connect4, pos, new_board)
  hor_count = count_indir(game, pos, (0, -1)) + count_indir(game, pos, (0, 1))
  ver_count = count_indir(game, pos, (1, 0)) + count_indir(game, pos, (-1, 0))
  d1_count = count_indir(game, pos, (1, -1)) + count_indir(game, pos, (-1, 1))
  d2_count = count_indir(game, pos, (1, 1)) + count_indir(game, pos, (-1, -1))

  if hor_count >= 3 || ver_count >= 3 || d1_count >= 3 || d2_count >= 3
    return -1
  else
    return 0
  end
end

function take_action(state::Connect4, action)
  new_board = copy(state.board)
  i = 1
  while i <= size(new_board, 1)
    if new_board[i, action] == 0x00
      new_board[i, action] = state.player
      break
    end
    i += 1
  end

  value = is_endgame(state, (i, action))
  
  return Connect4(new_board, next_player(state), value, value < 0)
end

