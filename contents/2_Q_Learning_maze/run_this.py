"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

from maze_env import Maze
from RL_brain import QLearningTable
import time


def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            # observation为每次红色方块移动的四个点的坐标
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            """
            observation_: 移动后的坐标 结束后为：terminal
            reward: 移动后的奖励
            done: 移动后是否结束
            """
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                print("episode:{}, observation:{}, action:{}, reward:{}, observation_:{}"
                      .format(episode, observation, action, reward, observation_))
                break

    # end of game
    print('game over')
    env.destroy()
    RL.q_table.to_excel(time.strftime("%Y-%m-%d %H-%M-%S.xlsx", time.localtime()))


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
