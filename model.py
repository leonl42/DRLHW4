import tensorflow as tf
import tensorflow_probability as tfp
class Actor(tf.keras.Model):

    def __init__(self, num_actions) -> None:
        super(Actor, self).__init__()

        
        self._conv1 = tf.keras.layers.Conv2D(24,kernel_size = (5,5),strides=(1,1),padding = "valid",activation = "relu",kernel_regularizer = "l1_l2")
        self._maxpool1 = tf.keras.layers.MaxPool2D(pool_size = (5,5),strides = (3,3),padding = "valid")
        self._conv2 = tf.keras.layers.Conv2D(24,kernel_size = (5,5),strides=(1,1),padding = "valid",activation = "relu",kernel_regularizer = "l1_l2")
        self._maxpool2 = tf.keras.layers.MaxPool2D(pool_size = (5,5),strides = (3,3),padding = "valid")
        self._conv3 = tf.keras.layers.Conv2D(24,kernel_size = (5,5),strides=(1,1),padding = "valid",activation = "relu",kernel_regularizer = "l1_l2")
        self._global = tf.keras.layers.GlobalMaxPool2D()

        # actor
        self._dense1 = tf.keras.layers.Dense(256,activation = "relu",kernel_regularizer = "l1_l2")
        self._dense2 = tf.keras.layers.Dense(128,activation = "relu",kernel_regularizer = "l1_l2")
        self._denseMu1 = tf.keras.layers.Dense(1,activation = "tanh",kernel_regularizer = "l1_l2")
        self._denseMu2 = tf.keras.layers.Dense(1,activation = "sigmoid",kernel_regularizer = "l1_l2")
        self._denseMu3 = tf.keras.layers.Dense(1,activation = "sigmoid",kernel_regularizer = "l1_l2")
        self._denseSigma1 = tf.keras.layers.Dense(1,activation = "sigmoid",kernel_regularizer = "l1_l2")
        self._denseSigma2 = tf.keras.layers.Dense(1,activation = "sigmoid",kernel_regularizer = "l1_l2")
        self._denseSigma3 = tf.keras.layers.Dense(1,activation = "sigmoid",kernel_regularizer = "l1_l2")

    @tf.function
    def state_embedding(self,state,training):
        x = self._conv1(state,training = training)
        x = self._maxpool1(x,training = training)
        x = self._conv2(x,training = training)
        x = self._maxpool2(x,training = training)
        x = self._conv3(x,training = training)
        x = self._global(x,training = training)

        return x

    @tf.function
    def get_dist_parameters(self, state, training):

        x = self.state_embedding(state,training)

        x = self._dense1(x,training = training)
        x = self._dense2(x,training = training)
        actions_normal_mu1 = 1.0*self._denseMu1(x,training = training)
        actions_normal_mu2 = 1.0*self._denseMu2(x,training = training)
        actions_normal_mu3 = 1.0*self._denseMu3(x,training = training)
        actions_normal_Sigma1 = 1.0*self._denseSigma1(x,training = training)
        actions_normal_Sigma2 = 1.0*self._denseSigma2(x,training = training)
        actions_normal_Sigma3 = 1.0*self._denseSigma3(x,training = training)

        return tf.concat([actions_normal_mu1,actions_normal_mu2,actions_normal_mu3],axis = -1),tf.concat([actions_normal_Sigma1,actions_normal_Sigma2,actions_normal_Sigma3],axis = -1)

    @tf.function
    def call(self, state,actions,training):

        actions_normal_mu,actions_normal_Sigma = self.get_dist_parameters(state,training)

        props = tf.TensorArray(tf.float32,size = tf.shape(state)[0])
        for batch_dim in range(tf.shape(state)[0]):
            dist = tfp.distributions.MultivariateNormalDiag(loc = actions_normal_mu[batch_dim],scale_diag = actions_normal_Sigma[batch_dim])
            
            action = actions[batch_dim]
            props = props.write(batch_dim,dist.log_prob(action))


        return props.stack()

    @tf.function
    def call_actions(self, state,training):
        
        actions_normal_mu,actions_normal_Sigma = self.get_dist_parameters(state,training)

        actions = tf.TensorArray(tf.float32,size = tf.shape(state)[0])
        props = tf.TensorArray(tf.float32,size = tf.shape(state)[0])
        for batch_dim in range(tf.shape(state)[0]):
            dist = tfp.distributions.MultivariateNormalDiag(loc = actions_normal_mu[batch_dim],scale_diag = actions_normal_Sigma[batch_dim])
            samples_val = dist.sample()

            actions = actions.write(batch_dim,samples_val)
            props = props.write(batch_dim,dist.log_prob(samples_val))



        return actions.stack(), props.stack()


class Critic(tf.keras.Model):

    def __init__(self) -> None:
        super(Critic, self).__init__()

        
        self._dense1 = tf.keras.layers.Dense(512,activation = "relu")
        self._dense2 = tf.keras.layers.Dense(256,activation = "relu")  
        self._dense3 = tf.keras.layers.Dense(1,activation = "linear")
    
    @tf.function
    def call(self,x,training):

        x = self._dense1(x,training = training)
        x = self._dense2(x,training = training)
        x = self._dense3(x,training = training)

        return x