import numpy as np
import gym
import pickle
import copy

from NEAT_Brain import Network
from NEAT_Config import Config
from NEAT_Helper import *

config = Config()
env = gym.make(Config.env_name)


#generate the initial population 
population = []
for i in range(config.num_policies):
    pop = Network()
    population.append(pop)

for episode in range(config.num_generations):
    Reward = np.zeros(config.num_policies)
    for member in range(config.num_policies):#test each member of the population 
        policy = population[member]
        for i in range(config.num_iterations):#play the game multiple times for each network 
            Reward[policy] += policy.playthrough(env)    

    Reward /= config.num_iterations
    print(episode, np.mean(Reward), np.max(Reward))
    l1, l2 = zip(*sorted(zip(Reward, population)))#sort based on score 
    
    #kill off the weak
    population = list(l2[int(config.percentage_killed*config.num_policies):])
    Reward = list(l1[int(config.percentage_killed*config.num_policies):])

    if (episode % config.checkpoint_freq == 0) and  (episode != 0):
        champ = population[-1]#currently only test the top score over the course of 100 episodes

        with open('Champion_Network.pk1', 'wb') as output:#saves the entire network
            pickle.dump(champ, output, pickle.HIGHEST_PROTOCOL)

        #take the average score over a large amount of game
        #if this score beats a set threshold, the network has 'solved' the environment 
        summed_reward = 0
        for i in range(config.episodes_to_solve):
            summed_reward += policy.playthrough(env)

        score = summed_reward/config.episodes_to_solve
        
        print 'Average score over ' + \
            str(config.episodes_to_solve) + ' episodes: ' + str(score)
        if (score > config.score_to_solve):
            print 'The game is solve!'
            break

    #refill the population with new children
    mutants = [] 
    for i in range(int(config.percentage_killed*config.num_policies)):
        #picks a policy to mutate based on the normalised score
        curr_pol = np.random.choice(population)# p = Reward/sum(Reward))
        new_pol = copy.deepcopy(curr_pol)
        mutate(new_pol)
        mutants.append(new_pol)
    population += mutants


