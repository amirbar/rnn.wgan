import tensorflow as tf
from config import FLAGS, BATCH_SIZE, LAMBDA
from model import get_generator, get_discriminator


class FisherGAN():
    """Implements fisher gan objective functions 
    Modeled off https://github.com/ethancaballero/FisherGAN/blob/master/main.py
    Tried to keep variable names the same as much as possible
    """

    def __init__(self, rho=1e-5):
        self._rho = rho
        self._alpha = tf.get_variable("fisher_alpha", [0], initializer=tf.zeros_initializer)

    def loss_d_g(self, disc_fake, disc_real, fake_inputs, real_inputs, charmap, seq_length, Discriminator):
        
        # Compared to WGAN, generator cost remains the same in fisher GAN
        gen_cost = -tf.reduce_mean(disc_fake)

        # Calculate Fisher GAN disc cost
        # NOTE that below two lines are the
        # only ones that need change. 
        # E_P and E_Q refer to Expectation over real and fake.

        E_Q_f = tf.reduce_mean(disc_fake)
        E_P_f = tf.reduce_mean(disc_real)
        E_Q_f2 = tf.reduce_mean(disc_fake**2)
        E_P_f2 = tf.reduce_mean(disc_real**2)

        constraint = (1 - (0.5*E_P_f2 + 0.5*E_Q_f2))

        # See Equation (9) in Fisher GAN paper
        # In the original implementation, they use a backward computation with mone (minus one)
        # To implement this in tensorflow, we simply multiply the objective
        # cost function by minus one.
        disc_cost = -1.0 * (E_P_f - E_Q_f + self._alpha * constraint - self._rho/2 * constraint**2)


        return disc_cost, gen_cost