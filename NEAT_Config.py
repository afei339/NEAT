import gym

class Config():
    env_name = 'MountainCar-v0'
    mode = 'discrete'

    if env_name == 'CartPole-v0' or env_name == 'MountainCar-v0' or env_name == 'Acrobot-v1':
        mode = 'discrete'
    elif env_name == 'MountainCarContinuous-v0' or env_name == 'Pendulum-v0':
        mode = 'continuous'

    game = gym.make(env_name)
    s_size = len(game.reset())
    if mode == 'discrete':
        a_size = game.action_space.n
    else:
        a_size = game.action_space.shape[0]
        a_bounds = [game.action_space.low, game.action_space.high]
        a_range = game.action_space.high - game.action_space.low

    sigma = 0.1
    percentage_killed = 0.5

    mut_weight_p = .9
    mut_connection_p = .3
    mut_node_p = .2
    mut_enabled_p = .3

    num_generations = 1000
    num_policies = 100
    num_iterations = 3
    checkpoint_freq = 5

    score_to_solve = -110
    episodes_to_solve = 100
