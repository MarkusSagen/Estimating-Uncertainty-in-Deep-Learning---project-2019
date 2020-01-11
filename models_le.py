import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.compat.v1.keras.backend as K
K.set_image_data_format('channels_last')
from tensorflow.compat.v1.keras import initializers
from tensorflow.compat.v1.keras.models import Sequential, Model
from tensorflow.compat.v1.keras.layers import (Input, Dense, Dropout, Flatten, Lambda, Activation, 
                                               Wrapper, Conv2D, AveragePooling2D, MaxPooling2D, InputSpec,
                                               Add, BatchNormalization, ZeroPadding2D)
from tensorflow.compat.v1.keras.initializers import glorot_uniform
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
# import sonnet as snt

def LeNet_base(input_shape, classes):
    """
    Toy model, determinstic baseline
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    # model.add(Dense(classes, activation='softmax'))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

def LeNet_dropout(input_shape, classes):
    """
    Use only minimalistic model to get some statistics for misclassifications
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    # This dropout layer stays active during testing phase
    model.add(Lambda(lambda x: K.dropout(x, level=0.25)))
    model.add(Dense(512, activation='relu'))

    model.add(Lambda(lambda x: K.dropout(x, level=0.5)))
    model.add(Dense(classes))
    model.add(Activation('softmax'))

    return model

def LeNet_concreteDropout(input_shape, classes, N):
    # initialize weight regularization params
    # N = len(X_train)
    wd = 1e-2 / N
    dd = 2. / N
    """
    Toy model, Concrete Dropout method
    """
    model = Sequential()
    model.add(SpatialConcreteDropout(Conv2D(32, kernel_size=(3, 3),
                     activation='relu'),
                     weight_regularizer=wd,
                     dropout_regularizer=dd,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialConcreteDropout(Conv2D(64, kernel_size=(3, 3),
                     activation='relu'),
                     weight_regularizer=wd,
                     dropout_regularizer=dd,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialConcreteDropout(Conv2D(128, kernel_size=(3, 3),
                     activation='relu'),
                     weight_regularizer=wd,
                     dropout_regularizer=dd,
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(ConcreteDropout(Dense(512, activation='relu'),
                              weight_regularizer=wd,
                              dropout_regularizer=dd))
    model.add(Dropout(0.5))
    model.add(ConcreteDropout(Dense(classes),
                              weight_regularizer=wd,
                              dropout_regularizer=dd))
    model.add(Activation('softmax'))
    
    return model

def LeNet_llconcreteDropout(input_shape, classes, N):
    # initialize weight regularization params
    # N = len(X_train)
    wd = 1e-2 / N
    dd = 2. / N
    """
    Toy model, Concrete Dropout method
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(ConcreteDropout(Dense(512, activation='relu'),
                              weight_regularizer=wd,
                              dropout_regularizer=dd))
    model.add(Dropout(0.5))
    model.add(ConcreteDropout(Dense(classes),
                              weight_regularizer=wd,
                              dropout_regularizer=dd))
    model.add(Activation('softmax'))
    
    return model

class ConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given input Dense layer.
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(ConcreteDropout(Dense(8), input_shape=(16)))
        # now model.output_shape == (None, 8)
        # subsequent layers: no need for input_shape
        model.add(ConcreteDropout(Dense(32)))
        # now model.output_shape == (None, 32)
    ```
    `ConcreteDropout` can be used with arbitrary layers which have 2D
    kernels, not just `Dense`. However, Conv2D layers require different
    weighing of the regulariser (use SpatialConcreteDropout instead).
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """

    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(ConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(ConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 2, 'this wrapper only supports Dense layers'
        input_dim = np.prod(input_shape[-1])  # we drop only last dim
        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * int(input_dim)
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 0.1

        unif_noise = K.random_uniform(shape=K.shape(x))
        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

class SpatialConcreteDropout(Wrapper):
    """This wrapper allows to learn the dropout probability for any given Conv2D input layer.
    ```python
        model = Sequential()
        model.add(ConcreteDropout(Conv2D(64, (3, 3)),
                                  input_shape=(299, 299, 3)))
    ```
    # Arguments
        layer: a layer instance.
        weight_regularizer:
            A positive number which satisfies
                $weight_regularizer = l**2 / (\tau * N)$
            with prior lengthscale l, model precision $\tau$ (inverse observation noise),
            and N the number of instances in the dataset.
            Note that kernel_regularizer is not needed.
        dropout_regularizer:
            A positive number which satisfies
                $dropout_regularizer = 2 / (\tau * N)$
            with model precision $\tau$ (inverse observation noise) and N the number of
            instances in the dataset.
            Note the relation between dropout_regularizer and weight_regularizer:
                $weight_regularizer / dropout_regularizer = l**2 / 2$
            with prior lengthscale l. Note also that the factor of two should be
            ignored for cross-entropy loss, and used only for the eculedian loss.
    """
    def __init__(self, layer, weight_regularizer=1e-6, dropout_regularizer=1e-5,
                 init_min=0.1, init_max=0.1, is_mc_dropout=True, data_format=None, **kwargs):
        assert 'kernel_regularizer' not in kwargs
        super(SpatialConcreteDropout, self).__init__(layer, **kwargs)
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.is_mc_dropout = is_mc_dropout
        self.supports_masking = True
        self.p_logit = None
        self.p = None
        self.init_min = np.log(init_min) - np.log(1. - init_min)
        self.init_max = np.log(init_max) - np.log(1. - init_max)
        self.data_format = 'channels_last' if data_format is None else 'channels_first'

    def build(self, input_shape=None):
        self.input_spec = InputSpec(shape=input_shape)
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        super(SpatialConcreteDropout, self).build()  # this is very weird.. we must call super before we add new losses

        # initialise p
        self.p_logit = self.layer.add_weight(name='p_logit',
                                            shape=(1,),
                                            initializer=initializers.RandomUniform(self.init_min, self.init_max),
                                            trainable=True)
        self.p = K.sigmoid(self.p_logit[0])

        # initialise regulariser / prior KL term
        assert len(input_shape) == 4, 'this wrapper only supports Conv2D layers'
        if self.data_format == 'channels_first':
            input_dim = input_shape[1] # we drop only channels
        else:
            input_dim = input_shape[3]

        weight = self.layer.kernel
        kernel_regularizer = self.weight_regularizer * K.sum(K.square(weight)) / (1. - self.p)
        dropout_regularizer = self.p * K.log(self.p)
        dropout_regularizer += (1. - self.p) * K.log(1. - self.p)
        dropout_regularizer *= self.dropout_regularizer * int(input_dim)
        regularizer = K.sum(kernel_regularizer + dropout_regularizer)
        self.layer.add_loss(regularizer)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def spatial_concrete_dropout(self, x):
        '''
        Concrete dropout - used at training time (gradients can be propagated)
        :param x: input
        :return:  approx. dropped out input
        '''
        eps = K.cast_to_floatx(K.epsilon())
        temp = 2. / 3.

        input_shape = K.shape(x)
        if self.data_format == 'channels_first':
            noise_shape = (input_shape[0], input_shape[1], 1, 1)
        else:
            noise_shape = (input_shape[0], 1, 1, input_shape[3])
        unif_noise = K.random_uniform(shape=noise_shape)

        drop_prob = (
            K.log(self.p + eps)
            - K.log(1. - self.p + eps)
            + K.log(unif_noise + eps)
            - K.log(1. - unif_noise + eps)
        )
        drop_prob = K.sigmoid(drop_prob / temp)
        random_tensor = 1. - drop_prob

        retain_prob = 1. - self.p
        x *= random_tensor
        x /= retain_prob
        return x

    def call(self, inputs, training=None):
        if self.is_mc_dropout:
            return self.layer.call(self.spatial_concrete_dropout(inputs))
        else:
            def relaxed_dropped_inputs():
                return self.layer.call(self.spatial_concrete_dropout(inputs))
            return K.in_train_phase(relaxed_dropped_inputs,
                                    self.layer.call(inputs),
                                    training=training)

def LeNet_vi_flipout(input_shape, classes, num_updates):
    """
    Variational inference by Flipout.
    """
    divergence_fn = lambda q, p, ignore: (tfd.kl_divergence(q, p)/num_updates)
    
    model = Sequential()
    model.add(tfpl.Convolution2DFlipout(
            32, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            kernel_divergence_fn=divergence_fn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tfpl.Convolution2DFlipout(
            64, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            kernel_divergence_fn=divergence_fn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(tfpl.Convolution2DFlipout(
            128, kernel_size=(3, 3),
            activation='relu',
            input_shape=input_shape,
            kernel_divergence_fn=divergence_fn))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    # model.add(Dense(classes, activation='softmax'))
#    model.add(Dense(classes))
    model.add(tfpl.DenseFlipout(classes))
    model.add(Activation('softmax'))

    return model

def LeNet_llvi(input_shape, classes):
    """
    Last layer variational inference.
    """

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dropout(0.25))
    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.5))
    # model.add(Dense(classes, activation='softmax'))
#    model.add(Dense(classes))
    model.add(tfpl.DenseFlipout(classes))
    model.add(Activation('softmax'))

    return model

"""
Bayesian back propagation - Described in 'Weight uncertainty in Neural Networks'
"""
'''
class BNNLayer(snt.AbstractModule):
    """
    Implementation of a linear Bayesian layer with n_inputs and n_outputs, and a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, init_mu=0.0, init_rho=0.0, name="BNNLayer"):
        super(BNNLayer, self).__init__(name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.w_mean = tf.Variable(init_mu*tf.ones([self.n_inputs, self.n_outputs]))
        self.w_rho = tf.Variable(init_rho*tf.ones([self.n_inputs, self.n_outputs]))
        self.w_sigma = tf.log(1.0 + tf.exp(self.w_rho))
        self.w_distr = tf.distributions.Normal(loc=self.w_mean, scale=self.w_sigma)

        self.b_mean = tf.Variable(init_mu*tf.ones([self.n_outputs]))
        self.b_rho = tf.Variable(init_rho*tf.ones([self.n_outputs]))
        self.b_sigma = tf.log(1.0 + tf.exp(self.b_rho))
        self.b_distr = tf.distributions.Normal(loc=self.b_mean, scale=self.b_sigma)

        self.w_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)
        self.b_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)


    def _build(self, inputs, sample=False):
        """
        Constructs the graph for the layer.
        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value
        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
        """
        if sample:
            w = self.w_distr.sample()
            b = self.b_distr.sample()
        else:
            w = self.w_mean
            b = self.b_mean

        z = tf.matmul(inputs,w) + b
        log_probs = tf.reduce_sum(self.w_distr.log_prob(w)) + tf.reduce_sum(self.b_distr.log_prob(b)) - tf.reduce_sum(self.w_prior_distr.log_prob(w)) - tf.reduce_sum(self.b_prior_distr.log_prob(b))

        return z, log_probs

class BNN_MLP(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """
    def __init__(self, n_inputs, n_outputs, hidden_units=[], init_mu=0.0, init_rho=0.0, activation=tf.nn.relu, last_activation=tf.nn.softmax, name="BNN_MLP"):
        super(BNN_MLP, self).__init__(name=name)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.last_activation = last_activation
        hidden_units = [n_inputs] + hidden_units + [n_outputs]

        self.layers = []
        for i in range(1, len(hidden_units)):
            self.layers.append( BNNLayer(hidden_units[i-1], hidden_units[i], init_mu=init_mu, init_rho=init_rho) )


    def _build(self, inputs, sample=False, n_samples=1, targets=None, loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y) ):
        """
        Constructs the MLP graph.
        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value
          n_samples: number of sampled networks to average output of the MLP over
          targets: target outputs of the MLP, used to compute the loss function on each sampled network
          loss_function: lambda function to compute the loss of the network given its output and targets.
        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
          avg_loss: `tf.Tensor` average loss across n_samples, computed using `loss_function'
        """
        log_probs = 0.0
        avg_loss = 0.0

        if not sample:
            n_samples = 1

        output = 0.0 ## avg. output logits
        for ns in range(n_samples):
            x = inputs
            for i in range(len(self.layers)):
                x, l_prob = self.layers[i](x, sample)
                if i == len(self.layers)-1:
                    x = self.last_activation(x)
                else:
                    x = self.activation(x)
                log_probs += l_prob
            output += x

            if targets is not None:
                if loss_function is not None:
                    loss = tf.reduce_mean(loss_function(x, targets), 0)
                    avg_loss += loss
        log_probs /= n_samples
        avg_loss /= n_samples
        output /= n_samples

        return output, log_probs, avg_loss
'''