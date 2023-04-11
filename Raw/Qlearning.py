
# %%
import gym
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np 
import random
import pickle
import time

#random.seed(0)
#np.random.seed(0)

# Параметры обучения
num_episodes = 100000
epsilon_decay = 0.999999
epsilon_min = 0.07
discount_factor = 0.9
learning_rate = 0.05
update_Q_matrix = False

# Параметры генератора состояния по видеофрейму
# Границы ринга в оригинальном фрейме
X_MIN = 35 
X_MAX = 177 
Y_MIN = 31
Y_MAX = 128
# Размерность ринга
X_LIM = 142 
Y_LIM = 97
# Цвета игроков
ORIG_COL0 = 332
ORIG_COL1 = 642
COL0 = 2
COL1 = 0
# Область уточнения головы
HX = 10
HY = 8
# Границы областей для определения примерного положения противника
STX = [40, 102]
STY = [24, 73]
# Децимация/границы для расстояния м/у игроками
DEC = 2
DLIMX = 12
DLIMY = 15


def pause():
    input_ret = input("Press the <ENTER> to continue or 'Q' to stop...")
    if (input_ret == 'q') or (input_ret == 'Q'):
        raise Exception('exit')

def recolor(x):
    if x == 0:
        return 0
    elif x == ORIG_COL0:
        return 1
    elif x == ORIG_COL1:
        return 2
    else:
        return 3

def cleanColor(x, a):
    return 1 if (x==a) else x

# Функция определения состояния игры по видеофрейму
# Состояние включает в себя:
# - Расстояние между игроками (dx, dy) - дискретизированное и клиппированное
# - Примерное положение противника на ринге (9 возможных позиций)
# Состояние преобразуется в одно целое число
def getState(observation):
    
    # Вырезаем область ринга (без границ) 
    observationReshaped = observation[X_MIN:X_MAX, Y_MIN:Y_MAX,:]
    observationDownscaled = observationReshaped[::1, ::1, :]

    # Переводим цвета в три значения (фон, игрок 1, игрок 2)
    img = np.sum(observationDownscaled, axis=2)
    img = np.vectorize(recolor)(img)

    # Находим барицентры фигур игроков 1 и 2 
    x0 = int(np.round(np.mean(np.nonzero(img == COL0)[0])))
    y0 = int(np.round(np.mean(np.nonzero(img == COL0)[1])))
    x1 = int(np.round(np.mean(np.nonzero(img == COL1)[0])))
    y1 = int(np.round(np.mean(np.nonzero(img == COL1)[1]))) 
    
    # Уточняем позицию головы игрока 1 
    # средняя точка, для которой цвет игрока макс. по линиям (x,:) и (:,y)
    if ((x0-HX)>=0 and (x0+HX)<X_LIM and (y0-HY)>=0 and (y0+HY)<Y_LIM):
        img1 = img[(x0-HX):(x0+HX), (y0-HY):(y0+HY)]
        a = np.sum((img1==COL0)*1, axis=1)
        x0 += int(np.mean(np.nonzero((a==np.max(a))*1)) - HX)
        b = np.sum((img1==COL0)*1, axis=0)
        y0 += int(np.mean(np.nonzero((b==np.max(b))*1)) - HY) 
    
    # Уточняем позицию головы игрока 2 (аналогично)
    if ((x1-HX)>=0 and (x1+HX)<X_LIM and (y1-HY)>=0 and (y1+HY)<Y_LIM):
        img2 = img[(x1-HX):(x1+HX), (y1-HY):(y1+HY)]
        a = np.sum((img2==COL1)*1, axis=1)
        x1 += int(np.mean(np.nonzero((a==np.max(a))*1)) - HX)
        b = np.sum((img2==COL1)*1, axis=0)
        y1 += int(np.mean(np.nonzero((b==np.max(b))*1)) - HY)    
    
    # Сокращаем размернось и вводим ограничения 
    dx = int(np.round((x1-x0)/DEC))
    dx = dx if dx<=DLIMX else DLIMX
    dx = dx if dx>=-DLIMX else -DLIMX
    dy = int(np.round((y1-y0)/DEC))
    dy = dy if dy<=DLIMY else DLIMY
    dy = dy if dy>=-DLIMY else -DLIMY

    # Флаг, который показывает, что позиции игроков поменялись
    oob = False if dy >= 0 else True

    # Примерная позиция противника на ринге (9 позиций на поле 3х3)
    zx1, zy1 = 0, 0
    if x1<STX[0]:
        zx1 = 1
    elif x1>=STX[1]:
        zx1 = 2
    if y1<STY[0]:
        zy1 = 1
    elif y1>=STY[1]:
        zy1 = 2

    # Собираем состояние в виде одного целого числа
    s = dx
    s += dy*(2*DLIMX+1)
    s += zy1*(2*DLIMY+1)*(2*DLIMX+1)
    s += zx1*(2*DLIMY+1)*(2*DLIMX+1)*3

    return s, oob

env = gym.make("Boxing-v0")

state_size = (2*DLIMX+1)*(2*DLIMY+1)*(3**2)
action_size = env.action_space.n
done = False

if True:
    print('Read Q-matrix from file')
    with open('Q5.pkl','rb') as f:
        Q, epsilon, n_start, rewards = pickle.load(f)
else:
    print('Initialize Q-matrix')    
    Q = np.zeros((state_size, action_size))
    epsilon = 1.0
    n_start = 0
    rewards = []

tic = time.perf_counter()

for n in range(n_start, num_episodes):
    observation = env.reset()
    state, oob = getState(observation)
    episode_reward = 0

    k = 0
    while True:
        #env.render(render_mode='human')
        k += 1

        if (np.random.rand() <= epsilon):
            # Рандомизация (exploratio-vs-exploitation)
            action = env.action_space.sample()
        else:
            # Добавляем случайный шум, чтобы при прочих равных выбор был случайным  
            noise = 10.0 * np.random.random((1, env.action_space.n)) / ((n+1)**2)
            action = np.argmax(Q[state, :] + noise)

        epsilon = epsilon*epsilon_decay if (epsilon > epsilon_min) else epsilon_min

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        new_state, oob = getState(observation)

        # Обновление Q-матрицы
        if update_Q_matrix:
            Qtarget = reward + discount_factor * np.max(Q[new_state, :])
            Q[state, action] = (1-learning_rate) * Q[state, action] + learning_rate * Qtarget

        state = new_state

        if done:
            toc = time.perf_counter()
            observation = env.reset()
            state, oob = getState(observation)
            print(f'Episode: {n+1}/{num_episodes}, Len: {k}, Reward: {episode_reward}, Eps: {np.round(epsilon,2)}, oob: {oob}, {toc - tic:0.0f} s')
            if (n+1)%50==0:
                reward_avg = np.convolve(rewards, np.ones(250)/250)
                reward_avg = reward_avg[:-250]
                plt.plot(rewards)
                plt.plot(reward_avg, 'r')
                plt.show()
            if (n+1)%250==0:
                print(f'Mean reward (last 250): {np.mean(rewards[-250::])}')
            break

    rewards.append(episode_reward)

env.close()

print(f'Mean reward: {np.mean(rewards)}')

#%%

with open('Q5.pkl','wb') as f:
    pickle.dump([Q, epsilon, n, rewards], f)

#%%
#plt.plot(rewards)




