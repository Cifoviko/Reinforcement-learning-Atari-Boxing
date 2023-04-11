#%%

from PIL import Image
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
from keras.optimizers import Adam
import keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

# Базовые настройки скрипта
ENV_NAME = 'Boxing-v0'
MODE = 'train'  # выбрать 'train' или 'test'
WEIGHTS = None

# Размерность вектора состояния (84x84 x 4)
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4


class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3  # (height, width, channel)
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # сжатие и перевод в градации серого
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.) # Клиппинг вознаграждения, как в Mnih et al (для boxing не важно)


# Инициализируем среду и получаем кол-во действий
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Строим модель (как у Mnih et al)
input_shape = (WINDOW_LENGTH,) + INPUT_SHAPE
model = Sequential()
if K.image_data_format() == 'channels_last':
    # (width, height, channels)
    model.add(Permute((2, 3, 1), input_shape=input_shape))
elif K.image_data_format() == 'channels_first':
    # (channels, width, height)
    model.add(Permute((1, 2, 3), input_shape=input_shape))
else:
    raise RuntimeError('Unknown image_data_format.')

# Топология сети:
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# КОнфигурируем память для experience replay
memory = SequentialMemory(limit=1000000, window_length=WINDOW_LENGTH)
processor = AtariProcessor()

# Конфигурируем policy (как у Mnih et al)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,
                              nb_steps=1000000)

# Конфигурируем агента (как у Mnih et al)
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])

if MODE == 'train':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    dqn.fit(env, callbacks=callbacks, nb_steps=1750000, log_interval=10000)

    # Сохраняем финальные веса
    dqn.save_weights(weights_filename, overwrite=True)

    # Тестируем алгоритм на 10 эпизодах
    dqn.test(env, nb_episodes=10, visualize=False)

elif MODE == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(ENV_NAME)
    if WEIGHTS:
        weights_filename = WEIGHTS
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)

