"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        """
        actions: 移动行为 'u', 'd', 'l', 'r' 上下左右
        learning_rate: 学习率
        reward_decay: 奖励衰减度
        e_greedy: 选择概率
        """
        self.actions = actions  # a list

        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        """
        observation: 为一个字符串 选择行为
        observation: 为每次红色方块移动的四个点的坐标
        """
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action 选择一行
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions 随机化，避免选择相同的值
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, s, a, r, s_):
        """
        s: 移动前的坐标列表，为一个字符串数据形式
        a: 移动行为，为一个索引
        r: 奖励机制
        s_: 移动后的坐标列表，为一个字符串数据形式
        """
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            # 进行奖励叠加
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update

    def check_state_exist(self, state):
        """
        state为每次红色方块移动的四个点的坐标
        并将state作为的
        """
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table._append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,# series的索引为DataFrame的列名
                    name=state, # 这里的name可以理解为列名，但是在追加到DataFrame末尾后，就变为index了
                )
            )