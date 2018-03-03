function trainer()
  logger("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")
  logger("=*=*=*=*=*=       NEW LOG       =*=*=*=*=*")
  logger("=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*")

  env = Game()

  ######## LOAD MEMORIES IF NECESSARY #####
  if Config.INITIAL_MEMORY_VERSION == 0
    memory = Memory(Config.MEMORY_SIZE)
  else
    # memory = 
    # "/run${pad(Config.INITIAL_RUN_NUMBER}, 4)/memory/memory${pad(Config.INITIAL_MEMORY_VERSION, 4)}.jld2"
  ######## LOAD MODEL IF NECESSARY ########
  ######## CREATE THE PLAYERS #############
  current_player = Agent(???)
  best_player = Agent(???)

  ######## MAIN LOOP ######################
  iteration = 0
  while true
    # New loop init
    iteration += 1
    # Self play
    play_result = play_matches(best_player, best_player, Config.EPISODES, Config.TURNS_UNTIL_TAU0, memory)
    memory = play_result.memory
    # clear_st!(memory)? Is it needed?
    if length(memory) >= Config.MEMORY_SIZE
      # RETRAINING
      replay!(current_player, memory)
      # TODO: may be save some checkpoints? Callbacks?
      # TODO: log some sampled info

      # TOURNAMENT
      play_result = play_matches(current_player, best_player, Config.EVAL_EPISODES, 0)
      if scores(play_result, current_player) > scores(play_result, best_player) * Config.SCORING_THRESHOLD
        best_player_version += 1
        # TODO: update best_player with current_player params
      end
    else
      # Log some info
    end
  end
end
