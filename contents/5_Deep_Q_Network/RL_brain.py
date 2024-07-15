"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        """
        n_actions: 行为长度
        n_features: 特征数
        learning_rate: 学习率
        reward_decay: 奖励衰减
        e_greedy: 随机选择
        replace_target_iter: 特征替换周期
        memory_size: 记忆长度
        batch_size: 批次大小
        e_greedy_increment: 贪婪增量
        output_graph: tensorflow日志输出
        """
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')

        # tf.assign 用于将评估网络（evaluation network）的参数值复制到目标网络（target network）的对应参数中
        """
        ref: 这是一个变量，表示要被赋值的目标变量
        value: 这是一个张量，表示要赋给目标变量的新值。它的形状和类型必须与目标变量兼容
        validate_shape (可选): 布尔值。如果为真，tf.assign 将验证 value 的形状是否与 ref 的形状匹配（默认行为）。如果为假，它允许 value 的形状与 ref 的形状不同。
        use_locking (可选): 布尔值。如果为真，赋值操作将在执行时使用锁，以确保操作的原子性
        name (可选): 操作的名称
        将value的值赋值给ref，并且需要在self.sess.run()之后才会被执行赋值成功！！！
        """
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # 创建一个 TensorFlow 会话 self.sess。会话是执行计算图的环境
        self.sess = tf.Session()

        # 如果 output_graph 为真，则创建一个 tf.summary.FileWriter，并将计算图写入 logs/ 目录。你可以通过运行 tensorboard --logdir=logs 来启动 TensorBoard 以可视化计算图
        # 注意：tf.summary.FileWriter 在 TensorFlow 1.0 中使用，在 TensorFlow 2.x 中被替换为 tf.summary.create_file_writer
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 运行全局变量初始化操作，初始化计算图中定义的所有变量
        self.sess.run(tf.global_variables_initializer())
        # 初始化一个空列表 self.cost_his，用于记录训练过程中的损失值（成本）。
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            """
            c_names: 变量集合名称，用于在后面指定变量集合
            n_l1: 第一层的神经元个数
            w_initializer: 权重初始化器，使用随机正态分布初始化权重
            b_initializer: 偏置初始化器，使用常量值0.1初始化偏置
            """
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            """
            w1: 第一层的权重变量
            b1: 第一层的偏置变量
            l1: 第一层的输出，使用ReLU激活函数
            """
            """
            创建变量作用域:
            通过 tf.variable_scope('l1') 创建一个名为 'l1' 的变量作用域。
            
            变量命名:
            在这个作用域内创建的所有变量的名称都会以 'l1' 为前缀。比如，在这个作用域内创建一个变量 w1，其完整名称将会是 'l1/w1'。这有助于在更复杂的模型中组织和区分变量。
            
            复用变量:
            变量作用域还可以用于变量共享（reuse）。当你需要在不同的地方使用同一组变量时，可以通过设置 reuse=True 来复用已经存在的变量。
            
            命名空间:
            变量作用域还提供了一个清晰的命名空间，以避免变量名称冲突。当你有多个不同的层或模块时，通过使用不同的变量作用域，可以确保它们之间的变量名称不会冲突。
            """
            with tf.variable_scope('l1'):
                """
                name ('w1'):
                这是变量的名称。在变量作用域内，这个名字是唯一的。通过这个名字可以在后续代码中引用这个变量。
                
                shape ([self.n_features, n_l1]):
                这是一个列表，定义了变量的形状。对于 w1 而言，它的形状是 [self.n_features, n_l1]，即一个二维张量，其中 self.n_features 是输入特征的数量，n_l1 是第一层神经元的数量。
                
                initializer (w_initializer):
                这是变量的初始化器，定义了如何初始化变量的初始值。在这个例子中，w_initializer 被设置为 tf.random_normal_initializer(0., 0.3)，表示用均值为0、标准差为0.3的正态分布随机初始化变量。
                
                collections (c_names):
                这是一个列表，指定变量应该被添加到哪些集合中。集合是一种组织和管理变量的方式。在这个例子中，c_names 包含两个集合名 ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]，意思是 w1 会被添加到这两个集合中。其中，'eval_net_params' 是自定义的集合名，而 tf.GraphKeys.GLOBAL_VARIABLES 是TensorFlow的全局变量集合。
                """
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            """
            w2: 第二层的权重变量
            b2: 第二层的偏置变量
            self.q_eval: 最终的Q值输出
            """
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # 按水平方向（列顺序）堆叠数组构成一个新的数组
        transition = np.hstack((s, [a, r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            """
            会去执行 with tf.variable_scope('eval_net'): 这其中的定义的代码
            将observation的值赋值给self.s，进行计算，计算过程为self.q_eval中的步骤，最后得到一个action_value值
            """
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            # 执行参数替换
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        # 获取一个self.batch_size大小的，其值为memory_size中随机获取的的一维数组
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        # 获取相应的sample_index数量的长度来获取存储中的数量 随机抽取值
        batch_memory = self.memory[sample_index, :]

        # 批量计算self.q_next self.q_eval这两个值，并且计算的数据来源为self.n_features的
        # 后self.n_features个与前self.n_features个
        q_next, q_eval = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={
                self.s_: batch_memory[:, -self.n_features:],  # fixed params
                self.s : batch_memory[:, :self.n_features],  # newest params
            })

        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()

        # 创建批次索引
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # 提取批次索引
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        # 提取批次索引奖励
        reward = batch_memory[:, self.n_features + 1]
        # 更新target值 反向传播更新值
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """

        # train eval network
        # 神经网络评估 计算其损失值，得到一个评估值
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s       : batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
