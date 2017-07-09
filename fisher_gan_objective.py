import tensorflow as tf

class FisherGAN():
    """Implements fisher gan objective functions 
    Modeled off https://github.com/ethancaballero/FisherGAN/blob/master/main.py
    Tried to keep variable names the same as much as possible

    To measure convergence, gen_cost should start at a positive number and decrease
    to zero. The lower, the better.

    Warning: in the very beginning of training, you may see the gen_cost rise. Please
    wait at least 5000 iterations and the gen_cost should start to lower. This 
    phenomena is due to the critic finding the appropriate wasserstein distance
    and then the generator adjusting for it.

    It is recommended that you use a critic iteration of 1 when using fisher gan
    """

    def __init__(self, rho=1e-5):
        tf.logging.warn("USING FISHER GAN OBJECTIVE FUNCTION")
        self._rho = rho
        # Initialize alpha (or in paper called lambda) with zero
        # Throughout training alpha is trained with an independent sgd optimizer
        # We use "alpha" instead of lambda because code we are modeling off of
        # uses "alpha" instead of lambda
        self._alpha = tf.get_variable("fisher_alpha", [], initializer=tf.zeros_initializer)

    def _optimize_alpha(self, disc_cost):
        """ In the optimization of alpha, we optimize via regular sgd with a learning rate
        of rho.

        This optimization should occur every time the discriminator is optimized because
        the same batch is used.

        Very crucial point --> We minimize the NEGATIVE disc_cost with our alpha parameter.
        This is done to enforce the Lipchitz constraint. If we minimized the positive disc_cost
        then our discriminator loss would drop to a very low negative number and the Lipchitz
        constraint would not hold.
        """

        # Find gradient of alpha with respect to negative disc_cost
        self._alpha_optimizer = tf.train.GradientDescentOptimizer(self._rho)
        self.alpha_optimizer_op = self._alpha_optimizer.minimize(-disc_cost, var_list=[self._alpha])
        return

    def loss_d_g(self, disc_fake, disc_real, fake_inputs, real_inputs, charmap, seq_length, Discriminator):
        
        # Compared to WGAN, generator cost remains the same in fisher GAN
        gen_cost = -tf.reduce_mean(disc_fake)

        # Calculate Lipchitz Constraint
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