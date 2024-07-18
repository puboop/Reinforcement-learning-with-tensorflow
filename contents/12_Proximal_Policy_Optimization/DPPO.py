"""
A simple version of OpenAI's Proximal Policy Optimization (PPO). [https://arxiv.org/abs/1707.06347]

Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.

The global PPO updating rule is adopted from DeepMind's paper (DPPO):
Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.3
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue

EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4  # parallel workers
GAMMA = 0.9  # reward discount factor
A_LR = 0.0001  # learning rate for actor
C_LR = 0.0002  # learning rate for critic
MIN_BATCH_SIZE = 64  # minimum batch size for updating PPO
UPDATE_STEP = 10  # loop update operation n-steps
EPSILON = 0.2  # for clipping surrogate objective
GAME = 'Pendulum-v1'
S_DIM, A_DIM = 3, 1  # state and action dimension


class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        """
        self.ctrain_op 是 Critic 的训练操作，使用 Adam 优化器来最小化 Critic 的损失
        """
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        # Critic 的输出，代表状态的值函数估计
        self.v = tf.layers.dense(l1, 1)
        # 经过折扣的奖励信号的占位符，用于计算优势（Advantage）
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        # 计算了每个样本的优势，即经过折扣的奖励减去 Critic 估计的值。
        self.advantage = self.tfdc_r - self.v
        # Critic 的损失函数，采用均方误差损失
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        # Critic 的训练操作，使用 Adam 优化器来最小化 Critic 的损失
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor网络通过 _build_anet 方法构建，返回策略 pi 和 oldpi，分别表示当前策略和旧策略
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        # 从当前策略 pi 中抽样动作的操作
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        # 将当前策略参数赋值给旧策略参数的操作，用于更新旧策略
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        # 动作的占位符，用于接收动作输入数据，维度为 [None, A_DIM]，其中 A_DIM 是动作空间的维度
        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        # 优势的占位符，用于接收优势输入数据，维度为 [None, 1]
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # 计算了当前策略和旧策略选择动作的概率比值
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        # 计算的 surrogate loss，即比例乘以优势
        surr = ratio * self.tfadv  # surrogate loss

        # Actor 的损失函数，采用 clipped surrogate objective 来最大化期望奖励
        self.aloss = -tf.reduce_mean(tf.minimum(  # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        # Actor 的训练操作，使用 Adam 优化器来最大化 Actor 的损失
        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()  # wait until get batch of data
                self.sess.run(self.update_oldpi_op)  # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]  # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()  # updating finished
                GLOBAL_UPDATE_COUNTER = 0  # reset counter
                ROLLING_EVENT.set()  # set roll-out available

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            # 全连接层，输入为 self.tfs（状态输入），输出为 200 个单元，使用 ReLU 激活函数
            # trainable 参数控制是否对该层进行训练
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            # 全连接层，输入为 l1（上一层的输出），输出为动作空间维度 A_DIM 的均值向量 mu
            # 使用了双曲正切（tanh）激活函数，并乘以2来缩放输出
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # 全连接层，输入为 l1，输出为动作空间维度 A_DIM 的标准差向量 sigma
            # 使用了 Softplus 激活函数，确保输出为正数
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            # 使用 mu 和 sigma 创建一个正态分布对象 norm_dist
            # 这个分布对象用于生成动作的概率分布
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        # 收集在变量作用域 name 下的所有全局变量
        # 这些变量包括了策略网络中所有的权重和偏置
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            # s = self.env.reset()
            s = self.env.reset()[0]
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():  # while global PPO is updating
                    ROLLING_EVENT.wait()  # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []  # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                # s_, r, done, _ = self.env.step(a)
                s_, r, done, _, __ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1  # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []  # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))  # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()  # stop collecting data
                        UPDATE_EVENT.set()  # globalPPO update

                    if GLOBAL_EP >= EP_MAX:  # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0:
                GLOBAL_RUNNING_R.append(ep_r)
            else:
                GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1] * 0.9 + ep_r * 0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP / EP_MAX * 100), '|W%i' % self.wid, '|Ep_r: %.2f' % ep_r, )


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()  # not update now
    ROLLING_EVENT.set()  # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]

    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()  # workers putting data in this queue
    threads = []
    for worker in workers:  # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()  # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update, ))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode');
    plt.ylabel('Moving reward');
    plt.ion();
    plt.show()
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]
