import numpy as np
import gym
import minedojo

import collections
import random
import cv2

import tensorflow as tf
import keras
from keras.models import Model
from keras import layers
from keras import backend as K
from keras import optimizers

import pickle

keras.backend.set_image_data_format('channels_last')


class MineAgent:
    def __init__(self, env):
        self.env = env
        self.num_actions = 45453                       
        self.epsilon = 1                               # epsilon propio
        self.epsilon_min = 0.1                         # valor de epsilon minimo
        self.decay_factor = 0.000018                   # valor por el que el epsilon es reducido en cada episodio
        self.discount_factor = 0.99                    # Factor con el que se descuentan futuras recompensa
        self.memory = collections.deque(maxlen=20000)  # Memoria para guardar experiencia
        self.learning_rate = 0.00025                   # Ratio de aprendizaje del a Neural Network
        self.image_width = 256                         # Ancho de la imagen del input
        self.image_height = 160                        # Altura de la imagen del input
        self.stack_depth = 4                           # Numero de imagenes que apilar
        self.model = self.create_CNN_model()           # Iniciar Neural network
        self.target_model = self.create_CNN_model()    # Iniciar el target model
        self.update_target_weights()  

    def create_CNN_model(self):

        input_shape = (self.stack_depth, self.image_height, self.image_width)
        actions_input = layers.Input((self.num_actions,), name = 'action_mask')

        frames_input = layers.Input(input_shape, name='input_layer')
        conv_1 = layers.Conv2D(32, (8,8), strides=4, padding ='same'\
        ,activation = 'relu', name='conv_1',kernel_initializer='glorot_uniform',bias_initializer='zeros')(frames_input)

        conv_2 = layers.Conv2D(64, (4,4), strides=2, padding='same', activation='relu',name='conv_2'\
           ,kernel_initializer='glorot_uniform',bias_initializer='zeros')(conv_1)

        conv_3 = layers.Conv2D(64, (3,3), strides=1, padding='same',name='conv_3', activation='relu'\
           ,kernel_initializer='glorot_uniform',bias_initializer='zeros')(conv_2)

        flatten_1 = layers.Flatten()(conv_3)

        dense_1 = layers.Dense(512, activation='relu', name='dense_1',
            kernel_initializer='glorot_uniform',bias_initializer='zeros')(flatten_1)
        output = layers.Dense(self.num_actions, activation='linear', name='output',
            kernel_initializer='glorot_uniform',bias_initializer='zeros')(dense_1)
        masked_output = layers.Multiply(name='masked_output')([output, actions_input])

        model = Model([frames_input, actions_input], [masked_output])
        optimizer = optimizers.Adam(lr = self.learning_rate)
        model.compile(optimizer, loss=tf.losses.huber)
        return model

    def huber_loss(self, y, q_value):
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        return loss

    def update_target_weights(self):
        self.target_model.set_weights(self.model.get_weights())

    def memorize(self, current_state, action, reward, next_state, done):
        self.memory.append([current_state, action, reward, next_state, done])

    def process_image(self, image):
        image = np.transpose(image, (1, 2, 0))
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #image = cv2.resize(image, (self.image_width, self.image_height))

        return image

    def add_zeros(b_number, size):
        while(len(b_number) < size):
            b_number = "0" + b_number
        return b_number

    def action_trans(act):
        #definimos lo que sera la accion nula
        res = np.array([0, 0, 0, 12, 12, 0, 0, 0])
        #llenamos los bits en caso que falten
        bin_act = MineAgent.add_zeros(act, 16)
        #separamos el binario en sus diferentes tramos
        act_int = int(bin_act[14:16], 2)
        act_cam_x = int(bin_act[4:9], 2)
        act_cam_y = int(bin_act[9:14], 2)
        act_jmp = int(bin_act[3:4], 2)
        act_mov = int(bin_act[0:3], 2)
        #si el valor que tiene la accion no nos interesa lo hacemos nulo
        if act_int > 2:
            act_int = 0
        if act_cam_x > 24:
            act_cam_x = 0
        if act_cam_y > 24:
            act_cam_y = 0
        if act_mov > 4:
            act_mov = 0
        #asiganmos los valores a la accion
        if(act_int == 2):
            res[5] = 3
        else:
            res[5] = act_int
        res[3] = act_cam_x
        res[4] = act_cam_y
        res[2] = act_jmp
        if(act_mov == 3):
            res[1] = 1
        elif(act_mov == 4):
            res[1] = 2
        else:
            res[0] = act_mov
        return res
            

    def greedy_action(self, current_state):
        current_state = np.float32(np.true_divide(current_state,255))
        action_mask = np.ones((1, self.num_actions))
        q_values = self.model.predict([current_state, action_mask])[0]
        greedy_action = np.argmax(q_values)
        greedy_action = MineAgent.action_trans(np.binary_repr(greedy_action))
        return greedy_action

    def calculate_targets(self, batch_size):

        current_states = []
        rewards = []
        actions = []
        next_states = []
        dones = []

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, next_state, done = sample
            current_states.append(state)
            rewards.append(reward)
            actions.append(action)
            next_states.append(next_state)
            dones.append(done)

        current_states = np.array(current_states)
        current_states = np.float32(np.true_divide(current_states,255))

        next_states = np.array(next_states)
        next_states = np.float32(np.true_divide(next_states,255))

        action_mask = np.ones((1, self.num_actions))

        action_mask = np.repeat(action_mask, batch_size, axis=0)

        next_Q_values = self.target_model.predict([next_states, action_mask])

        next_Q_values[dones] = 0

        targets = rewards + self.discount_factor * np.max(next_Q_values, axis=1)

        action_mask_current = self.get_one_hot(actions)

        return current_states, action_mask_current, targets

    def trans_single_action(action):
        #iniciamos los diferentes sectores de bits
        bin_mov = '000'
        bin_jmp = '0'
        bin_cam_x = '00000'
        bin_cam_y = '00000'
        bin_int = '00'
        #definimps la direccion del movimento
        if(action[0] > 0):
            bin_mov = np.binary_repr(action[0])
        elif(action[1] > 0):
            bin_mov = np.binary_repr(action[1] + 2)
        #definimos si salta o no
        if(action[2]== 1):
            bin_jmp = '1'
        #definimos los angulos de las camaras
        bin_cam_x = np.binary_repr(action[3])
        bin_cam_y = np.binary_repr(action[4])
        #definimos la interaccion con el mundo
        if(action[5] == 1):
            bin_int = '01'
        elif(action[5] == 3):
            bin_int = '10'
        #llenamos los bits en caso que falten
        bin_mov = MineAgent.add_zeros(bin_mov, 3)
        bin_cam_x = MineAgent.add_zeros(bin_cam_x, 5)
        bin_cam_y = MineAgent.add_zeros(bin_cam_y, 5)
        #juntamos los diferentes tramos en una fila de bits unica
        action =bin_mov + bin_jmp + bin_cam_x + bin_cam_y + bin_int 
        return action

    def get_one_hot(self, actions):
        actions = np.array(actions)
        actions2 = np.zeros(len(actions))
        for i in range(len(actions)):
            aux = MineAgent.trans_single_action(actions[i])
            actions2[i] =int(aux,2)
        one_hots = np.zeros((len(actions), self.num_actions))
        for i in range(self.num_actions):
            one_hots[:, i][np.where(actions2 == i)] = 1

        return one_hots

    def train_from_experience(self, states, action_mask, targets):
        labels = action_mask * targets[:,None]
        loss = self.model.train_on_batch([states, action_mask], labels)

    def save_model(self, name):
        self.model.save(name)

def cow_killer(env):
    #esta funcion borra las entidades una vez hemos completado un episodio
    #esto lo hace para no llegar al episodio 100 y estar rodeado 
    #de 100 criaturas en vez de 1 como aclaramos en el experimento
    for cmd in ["/kill @e[type=!player]", "/clear", "/kill @e[type=item]"]:
        env.env.unwrapped.execute_cmd(cmd)

env = minedojo.make(task_id="combat_sheep_plains_iron_armors_diamond_sword_shield", image_size=(160, 256))
agent = MineAgent(env)

stack_depth = 4

seq_memory = collections.deque(maxlen=stack_depth)

done = False

training = False

batch_size = 32

update_threshold = 5

save_threshold = 10

episodes =200 #1000001

time_steps = 300

collect_experience = agent.memory.maxlen - 5000

frame_skip = 4

ep_reward = []

for episode in range(1,episodes):
    seq_memory.clear()
    initial_state = env.reset()
    current_image = env.prev_obs["rgb"]
    frame = agent.process_image(current_image)
    frame = frame.reshape(1, frame.shape[0], frame.shape[1])
    current_state = np.repeat(frame, stack_depth, axis=0)
    seq_memory.extend(current_state)

    episode_reward = 0
    for time in range(time_steps):
        if time % frame_skip == 0:
            if training:
                agent.epsilon = agent.epsilon - agent.decay_factor
                agent.epsilon = max(agent.epsilon_min, agent.epsilon)

            if np.random.rand() <= agent.epsilon:
                if training:
                    print("rand:")
                aux_act = MineAgent.trans_single_action(env.action_space.sample())
                action = MineAgent.action_trans(aux_act)
            else:
                if training:
                    print("greedy:")
                action = agent.greedy_action(current_state.reshape(1, current_state.shape[0]\
                                   , current_state.shape[1], current_state.shape[2]))
        if training:
            print(action)
        next_pos, reward, done, _ = env.step(action)
        next_frame = env.prev_obs["rgb"]
        next_frame = agent.process_image(next_frame)
        seq_memory.append(next_frame)

        next_state = np.asarray(seq_memory)
        agent.memory.append([current_state, action, reward, next_state, done])

        current_state = next_state

        if len(agent.memory) == collect_experience:
            training = True
            print('Start training')

        if training:
            states, action_mask, targets = agent.calculate_targets(batch_size)
            agent.train_from_experience(states,action_mask, targets)

        episode_reward = episode_reward + reward


        if done: 
            break

    ep_reward.append([episode, episode_reward])

    print("episode: {}/{}, epsilon: {}, episode reward: {}"
      .format(episode, episodes, agent.epsilon, episode_reward))

    if training and (episode % update_threshold) == 0:
        print('Weights updated at episode:', episode)
        agent.update_target_weights()

    if training and (episode%save_threshold) == 0:
        print('Data saved at episode:', episode)
        agent.save_model('./train/DQN_CNN_model_{}.h5'.format(episode))
        pickle.dump(ep_reward, open('./train/rewards_{}.dump'.format(episode), 'wb'))

    cow_killer(env)

env.close()
