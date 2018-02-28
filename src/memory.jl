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

function commit_st!(memory::Memory)
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
