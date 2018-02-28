function trainer()
  ######## LOAD MEMORIES IF NECESSARY #####
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
    play_result = play_matches(best_player, best_player)
    memory = play_result.memory
    # clear_st!(memory)? Is it needed?
    if length(memory) >= Config.MEMORY_SIZE
      # RETRAINING
      replay!(current_player, memory)
      # TODO: may be save some checkpoints? Callbacks?
      # TODO: log some sampled info

      # TOURNAMENT
      play_result = play_matches(current_player, best_player)
      if scores(play_result, :current_player) > scores(play_result, :best_player) * Config.SCORING_THRESHOLD
        best_player_version += 1
        # TODO: update best_player with current_player params
      end
    else
      # Log some info
    end
  end
end
