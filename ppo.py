import gym
import tensorflow as tf
import random as rd
import numpy as np
import tensorflow_probability as tfp
from model import Actor, Critic
from tqdm import tqdm

def create_trajectory(model,render = False,length = 1000,epsilon = 0.7):
    """
    create trajectory given the model
    """
    env = gym.make("CarRacing-v1")
    env.action_space.seed(42)
    s_a_r_s = []

    observation, _ = env.reset(return_info=True)

    # normalize the image data
    observation = tf.cast(tf.convert_to_tensor(observation),tf.float32)
    observation = observation/127.5 -1.0

    for _ in range(length):

        if rd.randint(0,100)<=epsilon*100:
            with tf.device("/GPU:0"):
                action,_ = model.call_actions(tf.cast(tf.expand_dims(observation,axis = 0),tf.float32),False)
            action = tf.squeeze(action,axis = 0).numpy()
        else:
            action = [np.random.uniform(-1,1,1).astype(np.float32)[0],np.random.uniform(0,1,1).astype(np.float32)[0],np.random.uniform(0,1,1).astype(np.float32)[0]]
      
        new_observation, reward, done, _ = env.step(action)
            
        # normalize the image data
        new_observation = tf.cast(tf.convert_to_tensor(new_observation),tf.float32)
        new_observation = new_observation/127.5 -1.0

        s_a_r_s.append((observation,tf.convert_to_tensor(action),tf.convert_to_tensor(reward),new_observation))
        observation = new_observation
        if render:
            env.render()

        if done:
            observation, _ = env.reset(return_info=True)

            env.close()
            return s_a_r_s

    env.close()
    return s_a_r_s

def create_training_samples(trajectories, actor, critic, gamma):

    states = []
    actions = []
    rewards = []
    advantages = []
    for trajectory in trajectories:
        for s,a,r,new_s in trajectory:
            s = tf.cast(tf.convert_to_tensor(s),tf.float32)
            a = tf.cast(tf.convert_to_tensor(a),tf.float32)
            r = tf.cast(tf.convert_to_tensor(r),tf.float32)
            new_s = tf.cast(tf.convert_to_tensor(new_s),tf.float32)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            v_new_s = tf.squeeze(critic(actor.state_embedding(tf.expand_dims(new_s,axis = 0),False),False),axis = 0)
            v_s = tf.squeeze(critic(actor.state_embedding(tf.expand_dims(s,axis = 0),False),False),axis = 0)
            advantages.append(r + gamma * v_new_s - v_s)

    return tf.convert_to_tensor(states),tf.convert_to_tensor(actions),tf.convert_to_tensor(rewards),tf.squeeze(tf.convert_to_tensor(advantages),axis = -1)






actor = Actor(3)
critic = Critic()


optimizer_actor = tf.keras.optimizers.Adam(learning_rate = 0.0001)
optimizer_critic = tf.keras.optimizers.Adam(learning_rate = 0.0001)

gamma = 0.999
num_trajectories = 8
trajectory_length = 1200
visualize_every = 50
epsilon_ppo = 0.2
batch_size = 256

actor.load_weights('./weights_actor_critic/actor_run3')
critic.load_weights('./weights_actor_critic/critic_run3')

for epoch in range(1,1000):
    
    # check reward of current policy
    new_data = create_trajectory(actor, True if epoch%visualize_every == 0 else False,length = trajectory_length,epsilon = 1)
    reward = []
    for o in range(len(new_data)):
        s,a,r,new_s = new_data[o]
        reward.append(tf.math.pow(gamma,o)*r)
    print("round: ", epoch," average reward: ",tf.reduce_mean(reward))

    print("sampling trajectories...")
    new_trajectories = [create_trajectory(actor, False,length = trajectory_length,epsilon=1) for _ in tqdm(range(num_trajectories))]

    states,actions,rewards,Advantages = create_training_samples(new_trajectories,actor, critic,gamma)
    

    actor_theta_k = Actor(3)
    actor_theta_k(tf.expand_dims(states[0],axis = 0),tf.expand_dims(actions[0],axis = 0),False)
    actor_theta_k.set_weights(np.array(actor.get_weights(),dtype = object))
    
    for N in tqdm(range(4096)):
        random_indices = tf.random.uniform(shape=[batch_size],minval= 0,maxval=len(states),dtype=tf.int64)
        batch_states = tf.gather(states,random_indices)
        batch_actions = tf.gather(actions,random_indices)
        batch_Advantages = tf.gather(Advantages,random_indices)
        with tf.device("/GPU:0"):
            prob_theta_k = actor_theta_k(batch_states,batch_actions,False)

        with tf.GradientTape() as tape:

                with tf.device("/GPU:0"):
                    prob = actor(batch_states,batch_actions,True)
                indices_advantage_positive = tf.where(tf.math.less(-Advantages,0))
                indices_advantage_negative= tf.where(tf.math.less(Advantages,0))

                loss_advantage_positive = tf.minimum((tf.gather(prob,indices_advantage_positive)/tf.gather(prob_theta_k,indices_advantage_positive)),(1+epsilon_ppo))*tf.gather(Advantages,indices_advantage_positive)
                loss_advantage_negative = tf.maximum((tf.gather(prob,indices_advantage_negative)/tf.gather(prob_theta_k,indices_advantage_negative)),(1-epsilon_ppo))*tf.gather(Advantages,indices_advantage_negative)
                loss = -tf.reduce_sum(loss_advantage_negative+loss_advantage_negative)

        gradients = tape.gradient(loss,actor.trainable_weights)
        optimizer_actor.apply_gradients(zip(gradients, actor.trainable_weights))

    for N in tqdm(range(4096)):
        random_indices = tf.random.uniform(shape=[batch_size],minval= 0,maxval=len(states),dtype=tf.int64)
        batch_states = tf.gather(states,random_indices)
        batch_rewards = tf.gather(rewards,random_indices)

        with tf.device("/GPU:0"):
            batch_state_embeddings = actor.state_embedding(batch_states,False)
        with tf.GradientTape() as tape:
            with tf.device("/GPU:0"):
                v = critic(batch_state_embeddings,True)

            loss = tf.reduce_sum((v-batch_rewards)**2)


        gradients = tape.gradient(loss,critic.trainable_weights)
        optimizer_critic.apply_gradients(zip(gradients, critic.trainable_weights))
    

    actor.save_weights('./weights_actor_critic/actor_run3')
    critic.save_weights('./weights_actor_critic/critic_run3')


