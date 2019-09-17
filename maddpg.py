# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent
import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class MADDPG:
    def __init__(self, discount_factor, tau, batch_size):
        super(MADDPG, self).__init__()

        self.maddpg_agent = [DDPGAgent(24, 128, 128, 2, 52, 64, 64) for i in range(2)]
        
        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0
        self.batch_size = batch_size


    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, state_all_agents, noise):
        """get actions from all agents in the MADDPG object"""
        actions = [agent.act(state, noise) for agent, state in zip(self.maddpg_agent, state_all_agents[0])]
        return actions

    def target_act(self, state_all_agents, noise):
        """get target network actions from all the agents in the MADDPG object """

        target_actions = [self.maddpg_agent[i].target_act(state_all_agents[:, i, :], noise) for i in range(2)]
        return target_actions

    def soft_update(self, target, source, tau):
        """
        Perform DDPG soft update (move target params toward source based on weight
        factor tau)
        Inputs:
            target (torch.nn.Module): Net to copy parameters to
            source (torch.nn.Module): Net whose parameters to copy
            tau (float, 0 < x < 1): Weight factor for update
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def update(self, samples, agent_number, noise):
        """update the critics and actors of all the agents """
        
        state, action, reward, next_state, done = samples

        agent = self.maddpg_agent[agent_number]

        agent.critic_optimizer.zero_grad()

        reward = np.array(reward)
        done = np.array(done).astype(np.float)
        action = torch.from_numpy(np.array(action))
        state = torch.stack(state)

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(torch.stack(next_state), noise)
        target_actions = torch.cat(target_actions, dim = 1)
      
        target_critic_input = torch.cat((torch.stack(next_state).reshape(self.batch_size, -1), target_actions), dim = 1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)
            
       
        y = reward[:, agent_number]+ self.discount_factor * np.transpose(q_next.detach().numpy()) * (1 - done[:, agent_number])
        y = torch.from_numpy(np.transpose(y)).float()
        critic_input = torch.cat((state[:, 0, :], state[:, 1, :], action), dim = 1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y)
        critic_loss.backward()
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative

        q_input = [
            self.maddpg_agent[i].actor(state[:, i, :]) if i == agent_number else
            self.maddpg_agent[i].actor(state[:, i, :]).detach() for i in range(2)]

        q_input = torch.cat(q_input, dim = 1)

        # combine all the actions and observations for input to critic

        q_input2 = torch.cat((state[:, 0, :], state[:, 1, :], q_input), dim = 1)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()

        al = actor_loss.cpu().detach().item()
        cl = critic_loss.cpu().detach().item()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            self.soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            self.soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)


    





