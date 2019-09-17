from unityagents import UnityEnvironment

from maddpg import MADDPG
from ddpg import ReplayBuffer
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque


env = UnityEnvironment(file_name = "Tennis.app")


brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode = True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]


scores = [] 													 
scores_window = deque(maxlen = 100) 


def maddpg(n_episodes = 5000):


	#PARAMETERS:

	noise = 2
	batch_size = 256
	update_every = 1
	agent = MADDPG(discount_factor = 0.99, tau = 0.02, batch_size = batch_size)
	buff = ReplayBuffer(10000) 

	for i_episode in range(n_episodes):                                         
	    env_info = env.reset(train_mode = True)[brain_name]       
	    state = env_info.vector_observations 
	    state =  torch.from_numpy(np.array(state)).float().unsqueeze(0)
	    score = np.zeros(num_agents) 
	    t = 0

	    while True:
	        actions = agent.act(state, noise)
	        noise *= 0.9999
	        actions_array = torch.stack(actions).detach().numpy()
	        env_info = env.step(actions_array)[brain_name]  
	        next_state = env_info.vector_observations         
	        next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0)
	        reward = np.array(env_info.rewards).reshape(1, -1)
	        dones = np.array(env_info.local_done).reshape(1, -1)   
	        actions_array = actions_array.reshape(1, -1)               
	        buff.push((state, actions_array, reward, next_state, dones))

	        if len(buff) > batch_size and t % update_every == 0: 
		        for i in range(2):
		            samples = buff.sample(batch_size)
		            agent.update(samples, i, noise)
	        	agent.update_targets() 	
	     
	        t += 1						
	        score += reward[0]		                         
	        state = next_state                             
	        if np.any(dones):                                
	            break

	    scores_window.append(np.max(score))
	    scores.append(np.max(score))
	    print('\rEpisode {}\tAverage Score: {:.3f}'.format(i_episode, np.mean(scores_window)), end = "")

	    if i_episode % 100:
	    	for i in range(2):
	    		torch.save(agent.maddpg_agent[i].actor.state_dict(), 'bin/checkpoint_actor{}.pth'.format(i))
	    		torch.save(agent.maddpg_agent[i].critic.state_dict(), 'bin/checkpoint_critic{}.pth'.format(i))

	    if np.mean(scores_window) >= 0.5:
	    	print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.3f}'.format(i_episode - 100, np.mean(scores_window)))
	    	for i in range(2):
	    		torch.save(agent.maddpg_agent[i].actor.state_dict(), 'bin/actor{}_finished.pth'.format(i))
	    		torch.save(agent.maddpg_agent[i].critic.state_dict(), 'bin/critic{}_finished.pth'.format(i))
	    	break

if __name__ == '__main__':
	maddpg()

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(len(scores)), scores)
	plt.title('Progress of the agent over the episodes')
	plt.ylabel('Score')
	plt.xlabel('Episode #')
	plt.show()

