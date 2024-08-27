# const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::RockSamplePOMDP) = 1:pomdp.n_types+1
POMDPs.obsindex(pomdp::RockSamplePOMDP, o::Int) = o

function POMDPs.observation(pomdp::RockSamplePOMDP, a::Int, s::RSState)
    obs_tup = collect(1:pomdp.n_types+1) #Use SOneTo
    if a <= N_BASIC_ACTIONS
        # no obs
        return SparseCat(obs_tup, [zeros(pomdp.n_types)...,1.0]) # for type stability | Use @SArray for zeros and tuple
    else
        rock_ind = a - N_BASIC_ACTIONS 
        rock_pos = pomdp.rocks_positions[rock_ind]
        dist = norm(rock_pos - s.pos)
        efficiency = 0.5*(1.0 + exp(-dist*log(2)/pomdp.sensor_efficiency))
        other_efficiency = (1-efficiency)/(pomdp.n_types-1)
        rock_state = s.rocks[rock_ind]
        prob_tup = other_efficiency .* ones(pomdp.n_types+1)
        prob_tup[end] = 0.0
        prob_tup[rock_state] = efficiency
        return SparseCat(obs_tup, prob_tup)
    end
end