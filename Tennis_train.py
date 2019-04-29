from unityagents import UnityEnvironment
import numpy as np
#import pandas as pd
#import random
import torch
from collections import deque
import matplotlib.pyplot as plt
from ddpg_agent import Agent

env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space 
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])

def ddpg(n_episodes=2000, max_t=700):
    """ Deep Deterministic Policy Gradients Methods
    source: partially from https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal
    source: partially from my 1st project in this drlnd

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode

        reminder:
        epsilon = greedy factor for exploration vs. exploitation
        alpha = something like learnrate. defines the step size when an update occurs
        gamma = discount factor for rewards

    """    
    
    #scores = np.zeros(num_agents)      # list containing scores from each episode, per agent
    scores_window = deque(maxlen=100)  # last 100 scores
    agents = []                        # list of agents, depending on agents needed in the current environment
    scores_episodes = []
    for _ in range(num_agents):
        agents.append(Agent(state_size, 
                            action_size, 
                            random_seed=0,
                            buffer_size=int(1e6),  # replay buffer size (default: int(1e6))
                            batch_size=1024,       # minibatch size (default: 128)
                            gamma=0.98,            # discount factor (default: 0.99)
                            tau=1e-3,              # for soft update of target parameters (default: 1e-3)
                            lr_actor=1e-3,         # learning rate of the actor (default: 1e-3)
                            lr_critic=1e-4,        # learning rate of the critic (default: 1e-4)
                            #weight_decay=0.       # L2 weight decay (default: 3e-4)
                            weight_decay=1e-9      # L2 weight decay (default: 3e-4)
                            )
                       )

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations        
        for agent in agents:
            agent.reset()
        
        scores = np.zeros(num_agents) 
        
        for t in range(max_t):
            # interact with the enviornment
            actions = np.array([agents[a].act(states[a]) for a in range(num_agents)])
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            
            for a in range(num_agents):
                agents[a].step(t, states[a], actions[a], rewards[a], next_states[a], dones[a])
                
            states = next_states
            scores += rewards
            
            if t % 10:
                print('\rEpisode {}\tTimestep {}\tScore: {:.2f}\tmin: {:.2f}\tmax: {:.2f}'
                      .format(i_episode, t, np.mean(scores), np.min(scores), np.max(scores)), end="")
            
            if np.any(dones):
                break     
        score = np.mean(scores)
        scores_window.append(score)
        scores_episodes.append(score)
        #scores.append(score)
        
        print('\nEpisode {}\tAverage Score: {:.2f}\tScore: {:.2f}'.format(i_episode, np.mean(scores_window), score), end="")
        
        if np.mean(scores_window) >= 30.0 and i_episode > 99:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(Agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(Agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break
            
    return scores

scores = ddpg(n_episodes=2000, max_t=1000)

fig = plt.figure()
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results_plot.png')
plt.show()