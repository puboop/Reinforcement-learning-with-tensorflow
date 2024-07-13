# **************************************
# --*-- coding: utf-8 --*--
# @Time    : 2024-07-12
# @Author  : white
# @FileName: 强化学习-左右移动.py
# @Software: PyCharm
# **************************************
"""
A simple example for Reinforcement Learning using table lookup Q-learning method.
An agent "o" is on the left of a 1 dimensional world, the treasure is on the rightmost location.
Run this program and to see how the agent will improve its strategy of finding the treasure.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

B站视频：https://www.bilibili.com/video/av16921335/?p=5&t=3.297395&spm_id_from=333.1350.jump_directly&vd_source=81363306ecb49103823f13c17782ef94
"""

import numpy as np
import pandas as pd
import time

np.random.seed(2)  # reproducible

N_STATES = 6  # the length of the 1 dimensional world
ACTIONS = ['left', 'right']  # available actions
EPSILON = 0.9  # greedy police
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13  # maximum episodes
FRESH_TIME = 0.3  # fresh time for one move


def build_q_table(n_states, actions):
    # 创建一个states大小的DataFrame表，并且全为零
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions's name
    )
    # print(table)    # show table
    return table


def choose_action(state, q_table):
    """
    :param state:
    :param q_table:
    :return:
    """
    # 如何从DataFrame表中选择一个action（当前向哪个方向移动）
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]  # 选择一行
    # np.random.uniform()获取一个随机数，EPSILON当前绝大部分的选择
    # (state_actions == 0).all()当前行中全为0，为真
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        # 随机选择一个动作（向左，向右）
        action_name = np.random.choice(ACTIONS)
    else:  # act greedy
        # 否认则就以贪婪的模式来选择（始终选择最大的值）
        # idxmax()用于返回沿着指定轴的最大值第一次出现位置的索引
        action_name = state_actions.idxmax()  # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(S, A):
    """
    获取虚拟环境的反馈
    采取行动，获得下一个状态和奖励
    :param S: q_table的行索引
    :param A: action行为
    :return:
    """
    # This is how agent will interact with the environment
    if A == 'right':  # move right
        # 到达最右端
        if S == N_STATES - 2:  # terminate
            S_ = 'terminal'
            # 给予奖励
            R = 1
        else:
            # 继续向右移动
            S_ = S + 1
            # 不给予奖励
            R = 0
    else:  # move left
        R = 0
        # 已经到达最左端
        if S == 0:
            S_ = S  # reach the wall
        else:
            # 继续向左
            S_ = S - 1
    """
    S_有两个值，terminal，数字。这个终端，代表的是下一个状态是 宝藏
    terminal可不可以理解成已经找到”宝藏“了呢？
    R为奖励，0不奖励，1奖励
    """
    return S_, R


def update_env(S, episode, step_counter):
    """
    :param S: 当前移动到了第几步
    :param episode: 当前训练的轮数
    :param step_counter: 训练的总步数
    :return:
    """
    # 更新虚拟环境
    # This is how environment be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']  # '---------T' our environment
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            # 获取一个action
            A = choose_action(S, q_table)
            # 获取虚拟环境的反馈
            S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            q_predict = q_table.loc[S, A]  # 获取一个预测值
            # 进行奖励叠加
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()  # next state is not terminal
            else:
                q_target = R  # next state is terminal
                is_terminated = True  # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == "__main__":
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
