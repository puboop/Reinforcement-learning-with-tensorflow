"""
This part of code is the reinforcement learning brain, which is a brain of the agent.
All decisions are made in here.

Policy Gradient, Reinforcement Learning.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

# reproducible
np.random.seed(1)
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.95,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        # ep_obs 概率选取值
        # ep_as action选取值
        # ep_rs 奖励选取值
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # http://0.0.0.0:6006/
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        # 定义输入占位符
        with tf.name_scope('inputs'):
            # 第一层输入
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observations")
            # 存放动作
            self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
            # 存放动作对应的值
            self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")
        # fc1 tf.layers.dense全连接层
        layer = tf.layers.dense(
            inputs=self.tf_obs,  # 定义输入
            units=10,  # 神经元个数
            activation=tf.nn.tanh,  # tanh activation 定义激活函数 [-1, 1]
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),  # 权重初始化方法为均值为 0，标准差为 0.3 的正态分布
            bias_initializer=tf.constant_initializer(0.1),  # 偏置初始化方法为常数 0.1
            name='fc1'
        )
        # fc2
        all_act = tf.layers.dense(
            inputs=layer,  # 输入为上一层网络输入
            units=self.n_actions,  # 输出的神经元个数为特征个数
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='fc2'
        )

        # 对最终的输入结果再经过softmax处理（将值映射到0-1），并得到最终的输入结果
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')  # use softmax to convert to probability

        # 定义最终的损失函数
        with tf.name_scope('loss'):
            # sparse_softmax_cross_entropy_with_logits 计算稀疏Softmax交叉熵损失
            # logits 输入为最后一层神经网络
            # labels 标签参数为实际选择的动作
            # to maximize total reward (log_p * R) is to minimize -(log_p * R), and the tf only have minimize(loss)
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act,
                                                                          labels=self.tf_acts)  # this is negative log of chosen action
            # or in this way:
            # neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob)*tf.one_hot(self.tf_acts, self.n_actions), axis=1)

            # reduce_mean 计算张量平均值
            # 输入为上一步的损失值计算
            # neg_log_prob * self.tf_vt：负对数概率乘以对应的奖励
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt)  # reward guided loss
            # 得到最后的损失值

        with tf.name_scope('train'):
            # 使用AdamOptimizer（Adam ）优化器进行优化
            """
            minimize
            方法的主要功能是：
            计算梯度：根据提供的损失函数，计算模型参数的梯度
            应用梯度：将计算得到的梯度应用到模型参数上，以减少损失函数的值
            """
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # 选择一个动作
        # all_act_prob神经网络层，计算值
        # 替换tf_obs其中的变量值，并将observation添加一个维度，使其变为1行4列的二维数组
        """
        这里的作用，大致是，通过传入的observation来计算对应的action
        得到每个action的分布值，将每个action对应的值转换为概率（softmax输出，值在0-1之间）
        range函数将数组转为prob_weights的第二个维度的总长度转为列表，从列表中随机选择一个值
        其中每个值选择的概率为p指定的
        #######这里的选取方法，也是policy gradient的核心所在########
        """
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        # p=prob_weights.ravel()：ravel 方法将 prob_weights 转换为一维数组
        # p 参数指定了选择每个动作的概率，这里用 prob_weights 的值作为概率。
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        # 折扣和标准奖励回报计算
        discounted_ep_rs_norm = self._discount_and_norm_rewards()

        # train on episode
        # 神经网络训练与反向传播参数
        """
        self.tf_obs: np.vstack(self.ep_obs)：
        self.tf_obs 是输入的观察值（observations）的占位符
        self.ep_obs 是一个列表，存储了当前回合中的所有观察值
        np.vstack(self.ep_obs) 将 self.ep_obs 列表中的所有观察值垂直堆叠成一个二维数组，形状为 [None, n_obs]，其中 None 表示样本数量
        
        self.tf_acts: np.array(self.ep_as)：
        self.tf_acts 是输入的动作（actions）的占位符
        self.ep_as 是一个列表，存储了当前回合中的所有动作
        np.array(self.ep_as) 将 self.ep_as 列表转换为一个一维数组，形状为 [None, ]，其中 None 表示样本数量
        
        self.tf_vt: discounted_ep_rs_norm：
        self.tf_vt 是输入的奖励（rewards）的占位符
        discounted_ep_rs_norm 是当前回合中的折扣奖励（discounted rewards），通常经过标准化处理，形状为 [None, ]，其中 None 表示样本数量
        """
        self.sess.run(self.train_op, feed_dict={
            self.tf_obs : np.vstack(self.ep_obs),  # shape=[None, n_obs]
            self.tf_acts: np.array(self.ep_as),  # shape=[None, ]
            self.tf_vt  : discounted_ep_rs_norm,  # shape=[None, ]
        })

        self.ep_obs, self.ep_as, self.ep_rs = [], [], []  # empty episode data
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # 标准化奖励
        # discount episode rewards
        # 根据self.ep_rs的长度创建相应的全为0的ndarray数组
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        # 标准化奖励
        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
