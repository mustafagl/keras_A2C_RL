import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

#import scipy.signal

import time

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())


def hata_mat():
    print("hatali_hamle")


class Board:
    def __init__(self):
        self.board = np.array([
            ["a", "a", "a", "a", "a", "a", "a", "a"],
            ["a", "a", "a", "a", "a", "a", "a", "a"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""]
        ])

    def reset_board(self):
        self.board = np.array([
            ["a", "a", "a", "a", "a", "a", "a", "a"],
            ["a", "a", "a", "a", "a", "a", "a", "a"],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", ""]
        ])
        self.board = self.board.ravel()
        np.random.shuffle(self.board)

        self.board = self.board.reshape(8, 8)

    def move(self, source, target):
        if (self.board[source[1]][source[0]] == ""):
            # print("hata")
            agent.rew_history_white.append(-1.)
            print(source, target)

            return 0
        else:
            agent.rew_history_white.append(1 / 42.)
            print(self.board)

            self.board[target[1]][target[0]] = self.board[source[1]][source[0]]
            self.board[source[1]][source[0]] = ""

        print(source, target)

        return 1

    def find_in_board(self, source):
        indexes = np.where(self.board == "bb1")
        indexes = indexes[0][0], indexes[1][0]
        print(indexes)


board = Board()
layer = tf.keras.layers.StringLookup()
layer.adapt(board.board)
print(layer(board.board))


def network():
    data = layers.Input(shape=(8, 8), dtype=tf.string, batch_size=1)

    inp = layer(data)
    inp = layers.Embedding(64, 4)(inp)
    inp = layers.Flatten()(inp)

    actor_hid2 = layers.Dense(320, activation="relu")(inp)

    action = layers.Dense(64, activation="softmax")(actor_hid2)

    critic = layers.Dense(1)(actor_hid2)

    model = keras.models.Model(inputs=data, outputs=[action, critic])
    return model


class Agent(keras.Model):
    def __init__(self, white_network, black_network, eps):
        super(Agent, self).__init__()
        self.white_network = white_network
        self.black_network = black_network
        self.white_network_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        self.black_network_optimizer = keras.optimizers.Adam(learning_rate=1e-6)

        self.crit_history_white = []
        self.act_history_white = []
        self.rew_history_white = []
        self.gamma = 0.01
        self.crit_history_black = []
        self.act_history_black = []
        self.rew_history_black = []

        self.eps = eps

        self.huber_loss = keras.losses.Huber()

    def hamle_white(self):
        data = np.expand_dims(board.board, axis=0)

        act_result, crit_val = self.white_network(data)
        self.crit_history_white.append(crit_val[0, 0])
        action = np.random.choice(64, p=np.squeeze(act_result))
        self.act_history_white.append(tf.math.log(act_result[0, action]))
        source = action
        sourcex = source // 8
        sourcey = source % 8
        target = np.argmax(act_result) % 64
        # targetx = target // 8
        # targety = target % 8

        board.move((sourcex, sourcey), (0, 0))

    def hamle_black(self):
        inp = np.expand_dims(board.board, axis=0)
        start = time.time()
        result = self.black_network(inp)
        source = np.argmax(result) // 64
        sourcex = source // 8
        sourcey = source % 8
        target = np.argmax(result) % 64
        targetx = target // 8
        targety = target % 8
        return board.move((sourcex, sourcey), (targetx, targety))

    def play(self, epochs):
        for epoch in range(epochs):
            print(epoch)

            with tf.GradientTape() as tape:

                for i in range(10):
                    board.reset_board()
                    self.hamle_white()




                returns = []
                discounted_sum = 0

                for r in self.rew_history_white[::-1]:
                    print(r)
                    discounted_sum = r + self.gamma * discounted_sum
                    returns.insert(0, discounted_sum)

                returns = np.array(returns)
                returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
                returns = returns.tolist()
                returns.append(0)
                print(returns)



                history = zip(self.act_history_white, self.crit_history_white, returns)
                actor_losses = []
                critic_losses = []
                for act, crit, rew in history:
                    diff = rew - crit
                    print(diff)
                    actor_losses.append(-act * diff)  # actor loss
                    critic_losses.append(
                        self.huber_loss(tf.expand_dims(crit, 0), tf.expand_dims(rew, 0))
                    )

                loss_value = sum(actor_losses) + sum(critic_losses)
                print("actor loss: ", sum(actor_losses), "critic loss", sum(critic_losses))
                grads = tape.gradient(loss_value, self.white_network.trainable_variables)
                self.white_network_optimizer.apply_gradients(zip(grads, self.white_network.trainable_variables))

                # Clear the loss and reward history
                board.reset_board()
                self.act_history_white.clear()
                self.crit_history_white.clear()
                self.rew_history_white.clear()


eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0
net_white = network()
net_black = network()
net_white.summary()
agent = Agent(net_white, net_black, eps)
agent.play(100000)
