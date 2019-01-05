import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt


def train(agent, env, n_episodes=1000, max_t=300, score_window_size=100, max_score=30):

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    scores_deque = deque(maxlen=score_window_size)
    all_scores = []
    for i_episode in range(1, n_episodes + 1):
        agent.reset()
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)  # initialize the score
        for t in range(max_t):
            actions = agent.act(states)  # select an action
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done  # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            scores += env_info.rewards  # update the score
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        scores_deque.append(scores)
        all_scores.append(scores)

        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')

        if i_episode % score_window_size == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        elif np.mean(scores_deque) >= max_score:
            print(f'Task solved in {i_episode} episodes\tAverage Score: {np.mean(scores_deque):.2f}')
            #             torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            #             torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            return all_scores
        else:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

    return all_scores
