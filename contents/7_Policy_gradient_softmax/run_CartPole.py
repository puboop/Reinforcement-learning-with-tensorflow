"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 400  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

env = gym.make('CartPole-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
    output_graph=True,
)

for i_episode in range(3000):
    # 获取一个连续分布的array 其中为连续分布的特征值
    observation = env.reset()

    while True:
        if RENDER: env.render()
        # 根据连续分布选择一个动作
        action = RL.choose_action(observation)

        # 根据传入的action计算得知，当前的动作是否达到预期
        # observation_ 下一次observation的值
        # reward 奖励值
        # done 当前动作是否结束
        # info 动作信息
        observation_, reward, done, info = env.step(action)

        # 动作信息存储
        # observation 当前概率选取值
        # action 当前动作
        # reward 当前动作奖励
        RL.store_transition(observation, action, reward)

        # 当回合结束后，开始训练
        if done:
            # 总奖励
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            # 判断当前回合总奖励是否达到预期值，以决定是否熏染图形界面
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            if i_episode == 0:
                plt.plot(vt)    # plot the episode vt
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break

        observation = observation_
