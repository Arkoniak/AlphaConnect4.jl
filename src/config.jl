module Config

# EVALUATION
const SCORING_THRESHOLD = 1.3
const EVAL_EPISODES = 30

# SELF PLAY
const EPISODES = 30
const MCTS_SIMS = 50
const MEMORY_SIZE = 30_000
const TURNS_UNTIL_TAU0 = 10  # turns on which it starts playing deterministically
const CPUCT = 1
const EPSILON = 0.2
const ALPHA = 0.8

# RETRAINING
const BATCH_SIZE = 256
const EPOCHS = 1
const REG_CONST = 0.0001
const LEARNING_RATE = 0.1
const MOMENTUM = 0.9
const TRAINING_LOOPS = 10

# TODO: Model constants

end
