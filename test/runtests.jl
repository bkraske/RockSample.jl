using Random
using RockSample
using POMDPs
using POMDPTools
using Test
using Compose
using ParticleFilters
using StaticArrays

function test_state_indexing(pomdp::RockSamplePOMDP{K}, ss::Vector{RSState{K}}) where K
    for (i,s) in enumerate(states(pomdp))
        if s != ss[i]
            return false
        end
    end
    return true
end

@testset "state space" begin 
    pomdp = RockSamplePOMDP{3}()
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == length(pomdp)
    @test test_state_indexing(pomdp, ss)
    pomdp = RockSamplePOMDP{3}(map_size=(7, 10))
    state_iterator =  states(pomdp)
    ss = ordered_states(pomdp)
    @test length(ss) == length(pomdp)
    @test test_state_indexing(pomdp, ss)
end

@testset "convert_s" begin
    p = RockSamplePOMDP{3}()
    b0 = initialstate(p)
    s_test = rand(b0)
    v_s_test = convert_s(Vector{Float64}, s_test, p)
    s_back = convert_s(RSState, v_s_test, p)
    @test s_back == s_test
end

@testset "action space" begin 
    rng = MersenneTwister(1)
    pomdp = RockSamplePOMDP((5, 5), 3, rng)
    acts = actions(pomdp)
    b1 = ParticleCollection{RSState{3}}(
        RSState{3}[
            RSState{3}([1, 1], [0, 1, 0] .+ 1,1), 
            RSState{3}([1, 1], [1, 1, 1].+ 1,1)
        ], nothing)
    b2 = ParticleCollection{RSState{3}}(
        RSState{3}[
            RSState{3}([3, 1], [0, 1, 0].+ 1,1), 
            RSState{3}([3, 1], [1, 1, 1].+ 1,1)
        ], nothing)
    @test acts == ordered_actions(pomdp)
    @test length(acts) == length(actions(pomdp))
    @test length(acts) == RockSample.N_BASIC_ACTIONS + 3
    s = RSState{3}((3,1), (true, false, false),1)
    @test actions(pomdp, s) == actions(pomdp)
    s2 = RSState{3}((1,2), (true, false, false),1)
    @test length(actions(pomdp, s2)) == length(actions(pomdp)) - 1
    @test actionindex(pomdp, 1) == 1
    @test length(actions(pomdp, b1)) == length(actions(pomdp)) - 1
    @test actions(pomdp, b2) == actions(pomdp)
end

@testset "observation" begin 
    rng = MersenneTwister(1)
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    obs = observations(pomdp)
    @test obs == ordered_observations(pomdp)
    s0 = rand(rng, initialstate(pomdp))
    s0 = RSState{3}([1, 1], [2, 1, 1], 1)
    od = observation(pomdp, 1, s0)
    o = rand(rng, od)
    @test o == 3
    @inferred observation(pomdp, 6, s0)
    @inferred observation(pomdp, 1, s0)
    observation(pomdp, 6, s0)
    o = rand(rng, observation(pomdp, 6, s0))
    @test o == 2
    observation(pomdp, 7, s0)
    o = rand(rng, observation(pomdp, 7, s0))
    @test o == 1
    @test has_consistent_observation_distributions(pomdp)
end

@testset "reward" begin
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    rng = MersenneTwister(3)
    s = rand(rng, initialstate(pomdp))
    @show s
    @test reward(pomdp, s, 1, s) == pomdp.rock_rewards[1]
    @test reward(pomdp, s, 2, s) == pomdp.step_penalty
    s = RSState(RSPos(3,3), s.rocks, 1)
    @test reward(pomdp, s, 1, s) == pomdp.rock_rewards[2]
    @test reward(pomdp, s, 3, s) == pomdp.step_penalty
    @test reward(pomdp, s, 6, s) == pomdp.sensor_use_penalty
    @test reward(pomdp, s, 6, s) == 0.
    @test reward(pomdp, s, 2, s) == 0.
    s = RSState(RSPos(5,4), s.rocks, 1)
    sp = rand(rng, transition(pomdp, s, RockSample.BASIC_ACTIONS_DICT[:east]))
    @test reward(pomdp, s, RockSample.BASIC_ACTIONS_DICT[:east], sp) == pomdp.exit_reward
    
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1), step_penalty=-1., sensor_use_penalty=-5.)
    rng = MersenneTwister(3)
    s = rand(rng, initialstate(pomdp))
    @test reward(pomdp, s, 2, s) == -1.
    @test reward(pomdp, s, 6, s) == -5. - 1.
    @test reward(pomdp, s, 1, s) == pomdp.rock_rewards[1] - 1.
    s = RSState(RSPos(3,3), s.rocks,1)
    @test reward(pomdp, s, 1, s) == pomdp.rock_rewards[2] - 1.
end

@testset "simulation" begin 
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    rng = MersenneTwister(3)
    up = DiscreteUpdater(pomdp)

    # go straight to the exit
    policy = FunctionPolicy(s->RockSample.BASIC_ACTIONS_DICT[:east]) 
    hr = HistoryRecorder(rng=rng)
    b0 = initialstate(pomdp)
    s0 = rand(b0)
    rs_exit = solve(RSExitSolver(), pomdp)
    hist = simulate(hr, pomdp, policy, up, b0, s0)
    @test undiscounted_reward(hist) == pomdp.exit_reward
    @test discounted_reward(hist) ≈ discount(pomdp)^(n_steps(hist) - 1) * pomdp.exit_reward
    @test discounted_reward(hist) ≈ value(rs_exit, s0)
    @test value(rs_exit, pomdp.terminal_state) == 0.0

    # random policy
    policy = RandomPolicy(pomdp, rng=rng)
    hr = HistoryRecorder(rng=rng)
    hist = simulate(hr, pomdp, policy, up)
    @test n_steps(hist) > pomdp.map_size[1]
end

@testset "mdp/qmdp policy" begin
    pomdp = RockSamplePOMDP(5,5)
    @time solve(RSMDPSolver(), UnderlyingMDP(pomdp))
    @time solve(RSMDPSolver(), pomdp)
    @time solve(RSQMDPSolver(), pomdp)
end

@testset "rendering" begin 
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    s0 = RSState{3}((1,1), [2, 1, 2],1)
    display(render(pomdp, (s=s0, a=3)))
    b0 = initialstate(pomdp)
    display(render(pomdp, (s=s0, a=3, b=b0)))
end

@testset "constructor" begin
    @test RockSamplePOMDP() isa RockSamplePOMDP
    @test RockSamplePOMDP(rocks_positions=[(1,1),(2,2)]) isa RockSamplePOMDP{2}
    @test RockSamplePOMDP(7,8) isa RockSamplePOMDP{8}
    @test RockSamplePOMDP((13,14), 15) isa RockSamplePOMDP{15}
    @test RockSamplePOMDP((11,5), [(1,2), (2,4), (11,5)]) isa RockSamplePOMDP{3}
end

@testset "visualization" begin
    include("test_visualization.jl")
    test_initial_state()
    test_particle_collection()
end

@testset "transition" begin
    rng = MersenneTwister(1)
    pomdp = RockSamplePOMDP{3}(init_pos=(1,1))
    s0 = rand(rng, initialstate(pomdp))
    @test s0.pos == pomdp.init_pos
    d = transition(pomdp, s0, 2) # move up
    sp = rand(rng, d)
    spp = rand(rng, d)
    @test spp == sp
    @test sp.pos == [1, 2]
    @test sp.rocks == s0.rocks
    s = RSState{3}((pomdp.map_size[1], 1), s0.rocks, 1)
    d = transition(pomdp, s, 3) # move right
    sp = rand(rng, d)
    @test isterminal(pomdp, sp)
    @test sp == pomdp.terminal_state
    @inferred transition(pomdp, s0, 3)
    @inferred rand(rng, transition(pomdp, s0, 3))
    @test has_consistent_transition_distributions(pomdp)
    include("../src/transition.jl")
    include("../src/actions.jl")
    # test next_pose for each action
    s = RSState{3}((1, 1), s0.rocks, 1)
    # sample
    @test next_position(s, BASIC_ACTIONS_DICT[:sample]) == s.pos
    # north
    @test next_position(s, BASIC_ACTIONS_DICT[:north]) == [1, 2]
    # east
    @test next_position(s, BASIC_ACTIONS_DICT[:east]) == [2, 1]
    # south
    s = RSState{3}((1, 2), s0.rocks, 1)
    @test next_position(s, BASIC_ACTIONS_DICT[:south]) == [1, 1]
    # west
    s = RSState{3}((2, 1), s0.rocks, 1)
    @test next_position(s, BASIC_ACTIONS_DICT[:west]) == [1, 1]
    # test sense
    for i in 1:length(s.rocks)
        @test next_position(s, N_BASIC_ACTIONS+i) == s.pos
    end
end

@testset "Terminal States" begin
    pomdp1 = RockSamplePOMDP()
    for (i,s) in enumerate(ordered_states(pomdp1))
        @test i == stateindex(pomdp1,s)
    end
    phys_states = ordered_states(pomdp1)[findall(x->x.step==(pomdp1.horizon-1),ordered_states(pomdp1))]
    for s in phys_states
        if !isterminal(pomdp1,s)
            for a in actions(pomdp1)
                sps = support(transition(pomdp1,s,a))
                for sp in sps
                    @test isterminal(pomdp1,sp) == true
                end
            end
        end
    end
end

@testset "Horizon Incrementing" begin
    pomdp1 = RockSamplePOMDP()
    for s in ordered_states(pomdp1)
        if !isterminal(pomdp1,s)
            for a in actions(pomdp1)
                for sp in support(transition(pomdp1,s,a))
                    if !isterminal(pomdp1,sp)
                        @test s.step + 1 == sp.step
                    end
                end
            end
        end
    end
end

@testset "3+ Rocktypes" begin
    rock_vals = [-10.,10.,5.0]
    pomdp1 = RockSamplePOMDP(;rock_types = [1,2,3],rock_rewards = rock_vals)
    for (i,rock) in enumerate(pomdp1.rocks_positions)
        @test reward(pomdp1,RSState{3}(rock, [1, 2, 3],1),RockSample.BASIC_ACTIONS_DICT[:sample]) == rock_vals[i]
        @test isa(transition(pomdp1,RSState{3}(rock, [2, 2, 3],1),RockSample.BASIC_ACTIONS_DICT[:sample]),Deterministic)
        @test support(transition(pomdp1,RSState{3}(rock, [2, 2, 3],1),RockSample.BASIC_ACTIONS_DICT[:sample]))[1].rocks[i] == 1
        @show observation(pomdp1,RockSample.N_BASIC_ACTIONS+i,RSState{3}(rock, [1, 2, 3],1))
        @test argmax(observation(pomdp1,RockSample.N_BASIC_ACTIONS+i,RSState{3}(rock, [1, 2, 3],1)).probs) == i
    end
    @inferred initialstate(pomdp1)
end