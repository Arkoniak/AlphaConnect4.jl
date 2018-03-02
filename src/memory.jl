using StatsBase: sample

struct MemoryChunk{T}
  state::T
  id::Int
  av::Float64
  player_turn::Int
end

struct Memory{T}
  st_memory::Vector{T}
  lt_memory::Vector{T}
  memory_size::Int
end

function Memory(memory_size)
  Memory(
    Vector(),
    Vector(),
    memory_size
  )
end

function clear_st!(memory::Memory)
end

function clear_lt!(memory::Memory)
end

function commit_st!(memory::Memory, identities, state, action_values)
  for r in identities(state, action_values???)
    mem_chunk = MemoryChunk(r[1], r[1].id, r[2], r[1].player_turn)
    push!(memory.st_memory, mem_chunk)
  end
end

function commit_lt!(memory::Memory)
  for elem in memory.st_memory
    push!(memory.lt_memory, elem)
  end
  clear_st!(memory)
end

function length(memory::Memory)
  length(memory.lt_memory)
end

function get_minibatch(memory)
  sample(memory.lt_memory, min(Config.BATCH_SIZE, length(memory.lt_memory)), replace = false) 
end
