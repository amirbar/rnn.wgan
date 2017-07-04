import tensorflow as tf
from config import FLAGS, BATCH_SIZE, LAMBDA
from model import get_generator, get_discriminator


class FisherGAN():
    """Implements fisher gan objective functions 
    Modeled off https://github.com/ethancaballero/FisherGAN/blob/master/main.py
    Tried to keep variable names the same as much as possible


    To measure convergence, gen_cost should start at zero and decrease
    to a negative number. The lower, the better.

    It is recommended that you use a critic iteration of 1 when using fisher gan
    """

    def __init__(self, rho=1e-5):
        tf.logging.warn("USING FISHER GAN OBJECTIVE FUNCTION")
        self._rho = rho
        # Initialize alpha (or in paper called lambda) with zero
        # Throughout training alpha is trained with an independent sgd optimizer
        self._alpha = tf.get_variable("fisher_alpha", [], initializer=tf.zeros_initializer)

    def _optimize_alpha(self, disc_cost):
        """ In the optimization of alpha, we optimize via regular sgd with a learning rate
        of rho.
        This should occur every time the discriminator is optimized. 

        Very crucial point --> We minimize the NEGATIVE disc_cost with our alpha parameter.
        This is done to enforce the Lipchitz constraint.
        """

        # first find alpha gradient
        self._alpha_optimizer = tf.train.GradientDescentOptimizer(self._rho)
        self.alpha_optimizer_op = self._alpha_optimizer.minimize(-disc_cost, var_list=[self._alpha])
        return

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

        # calculate optimization op for alpha
        self._optimize_alpha(disc_cost)

        return disc_cost, gen_cost