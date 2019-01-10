import numpy as np
from collections import deque
import torch
import matplotlib.pyplot as plt
import time
import os
import pickle


def train(agent, env, n_episodes=1000, score_window_size=100, print_every=50, max_score=None, damp_exploration_noise=False):
    task_solved = False

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    scores_deque = deque(maxlen=score_window_size)
    all_scores = []
    all_avg_scores = []
    all_std = []

    start = time.time()
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        states = env_info.vector_observations  # get the current state
        scores = np.zeros(num_agents)  # initialize the score
        while True:
            if damp_exploration_noise:
                damping = (40 - np.mean(scores))/40
                actions = agent.act(states, noise_damping=damping)  # select an action
            else:
                actions = agent.act(states)  # select an action

            env_info = env.step(actions)[brain_name]  # send all actions to the environment

            next_states = env_info.vector_observations  # get next state
            rewards = env_info.rewards  # get reward
            dones = env_info.local_done  # see if episode finished

            agent.step(states, actions, rewards, next_states, dones)

            scores += rewards  # update the score
            states = next_states  # roll over states to next time step
            if np.any(dones):  # exit loop if episode finished
                break

        scores_deque.append(np.mean(scores))
        all_scores.append(scores)
        all_avg_scores.append(np.mean(scores_deque))
        all_std.append(np.std(scores_deque))

        if np.mean(scores_deque) >= max_score and not task_solved:
            print(f'\nTask solved in {i_episode} episodes\tAverage Score: {np.mean(scores_deque):.2f}')
            task_solved = True
        elif i_episode == n_episodes:
            print('')
        elif i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
        else:
            end = time.time()
            print('\rEpisode {}\tAverage Score: {:.2f}\tElapsed Time{:.2f}'.format(i_episode, np.mean(scores_deque), end-start), end="")
            start = end

    save_state(agent, all_avg_scores, all_scores, all_std, i_episode, scores_deque)

    return all_scores, all_avg_scores, all_std


def save_state(agent, all_avg_scores, all_scores, all_std, i_episode, scores_deque):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    agent_type = agent.__class__.__name__
    folder_name = agent_type + '-' + f'{np.mean(scores_deque):.2f}' + '-' + str(i_episode) + '-' + timestr
    save_path = './checkpoints/' + folder_name
    os.makedirs(save_path, exist_ok=True)
    torch.save(agent.actor_local.state_dict(), save_path + '/checkpoint_actor.pth')
    torch.save(agent.critic_local.state_dict(), save_path + '/checkpoint_critic.pth')
    pickle.dump(all_scores, open(save_path + "/scores.p", "wb"))
    pickle.dump(all_avg_scores, open(save_path + "/avg_scores.p", "wb"))
    pickle.dump(all_std, open(save_path + "/std.p", "wb"))
    plot_scores(all_scores, all_avg_scores, all_std, out_file=save_path + "/training_plot.pdf")


def plot_scores(scores, avgscores, std, out_file=''):

    if out_file:
        was_interactive = plt.isinteractive()
        plt.ioff()

    scores = np.squeeze(np.array(scores))
    avgscores = np.array(avgscores)
    std = np.array(std)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, color='blue', alpha=0.3)

    plt.plot(np.arange(1, len(scores) + 1), avgscores, 'b--')

    min_error = avgscores - 2 * std
    max_error = avgscores + 2 * std

    plt.fill_between(np.arange(1, len(scores) + 1), min_error, max_error, color='blue', alpha=0.1)
    plt.ylabel('Score')
    plt.xlabel('Episode #')

    if not out_file:
        plt.show()
    else:
        plt.savefig(out_file)
        plt.close(fig)
        plt.interactive(was_interactive)


def demo(agent, env):
    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)

    states = env_info.vector_observations                  # get the current state
    scores = np.zeros(num_agents)                          # initialize the score
    while True:
        actions = agent.act(states, add_noise=False)       # select an action

        env_info = env.step(actions)[brain_name]           # send all actions to the environment
        next_states = env_info.vector_observations         # get next state
        rewards = env_info.rewards                         # get reward
        dones = env_info.local_done                        # see if episode finished

        scores += rewards                                  # update the score
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))


if __name__ == '__main__':
    from unityagents import UnityEnvironment
    import numpy as np
    from ddpg_agent import DDPGAgent
    # from reacher import *
    import random as rand

    import matplotlib.pyplot as plt
    #%matplotlib inline

    env = UnityEnvironment(file_name='../resources/Reacher_Linux/Reacher.x86_64', worker_id=rand.randint(1, 100))

    # get the default brain and other environment data
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset environment and get task information
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    agent = DDPGAgent(state_size, action_size, random_seed=0)

    scores, std = train(agent, env, n_episodes=500, score_window_size=100, max_score=30)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores, color='blue')
    plt.fill_between(range(scores), scores - 2 * std, scores + 2 * std, color='blue', alpha=0.2)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    env.close()
