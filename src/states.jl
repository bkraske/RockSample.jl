function POMDPs.stateindex(pomdp::RockSamplePOMDP{K}, s::RSState{K}) where K
    if s == pomdp.terminal_state
        return length(pomdp)
    end
    rocks_dim = @SVector fill(1:pomdp.n_types, K)
    nx, ny = pomdp.map_size
    return LinearIndices((1:nx, 1:ny, rocks_dim..., 1:pomdp.horizon))[s.pos...,s.rocks..., s.step]
end

function state_from_index(pomdp::RockSamplePOMDP{K}, si::Int) where K
    if si == length(pomdp)
        return pomdp.terminal_state
    end
    rocks_dim = @SVector fill(pomdp.n_types, K)
    nx, ny = pomdp.map_size
    s = CartesianIndices((nx, ny, rocks_dim...,pomdp.horizon))[si]
    pos = RSPos(s[1], s[2])
    rocks = SVector{K, Int}(s.I[3:(K+2)])
    step = s.I[end]
    return RSState{K}(pos, rocks, step)
end

# the state space is the pomdp itself
POMDPs.states(pomdp::RockSamplePOMDP) = pomdp

Base.length(pomdp::RockSamplePOMDP) = pomdp.map_size[1]*pomdp.map_size[2]*pomdp.n_types^length(pomdp.rocks_positions)*pomdp.horizon + 1

# we define an iterator over it
function Base.iterate(pomdp::RockSamplePOMDP, i::Int=1)
    if i > length(pomdp)
        return nothing
    end
    s = state_from_index(pomdp, i)
    return (s, i+1)
end

function POMDPs.initialstate(pomdp::RockSamplePOMDP{K}) where K
    probs = normalize!(ones(2^K), 1)
    states = Vector{RSState{K}}(undef, 2^K)
    for (i,rocks) in enumerate(Iterators.product(ntuple(x->collect(1:pomdp.n_types), K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks),1)
    end
    return SparseCat(states, probs)
end
