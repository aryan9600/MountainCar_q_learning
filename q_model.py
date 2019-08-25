import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

learning_rate = 0.1
discount = 0.95
episodes = 20000

show_every = 500
stats_every = 100

DISCRETE_OS_SIZE = [40]*len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILOND_DECAYING = episodes//2

epsilon_decay_value = epsilon/(END_EPSILOND_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2, high=0, size=DISCRETE_OS_SIZE+[env.action_space.n])

ep_rewards = []
agg_ep_rewards = {'ep':[], 'avg':[], 'min':[], 'max':[]}

def get_discrete_state(state):
	discrete_state = (state - env.observation_space.low)/discrete_os_win_size
	return tuple(discrete_state.astype(np.int))

for episode in range(episodes):

	episode_reward = 0

	if episode%show_every==0:
		print(episode)
		render = True
	else:
		render = False

	discrete_state = get_discrete_state(env.reset())

	done = False

	while not done:

		if np.random.random() > epsilon:
			action = np.argmax(q_table[discrete_state])

		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		episode_reward += reward
		new_discrete_state = get_discrete_state(new_state)
		if render:
			env.render()
		
		if not done:
			max_future_q = np.max(q_table[new_discrete_state])
			current_q = q_table[discrete_state + (action,)]
			new_q = (1-learning_rate)*current_q + learning_rate*(reward + discount*max_future_q)
			q_table[discrete_state+(action,)] = new_q

		elif new_state[0]>=env.goal_position:
			q_table[discrete_state+(action,)] = 0
			print(f'We made it at {episode}')
		
		discrete_state = new_discrete_state

		if END_EPSILOND_DECAYING >= episode >= START_EPSILON_DECAYING:
			epsilon -= epsilon_decay_value

		ep_rewards.append(episode_reward)
		if not episode%stats_every:
			average_reward = sum(ep_rewards[-stats_every:])/stats_every
			agg_ep_rewards['ep'].append(episode)
			agg_ep_rewards['avg'].append(average_reward)
			agg_ep_rewards['min'].append(min(ep_rewards[-stats_every:]))
			agg_ep_rewards['max'].append(max(ep_rewards[-stats_every:]))
			print(f'Episode: {episode:>5d}, average_reward: {average_reward:>4.14f},current epsilon: {epsilon:>1.2f}')

		if episode%10==0:
			np.save(f'q_tables/{episode}-q_table.npy', q_table)

env.close()

plt.plot(agg_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
plt.plot(agg_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
plt.plot(agg_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
plt.legend(loc=4)
plt.show()