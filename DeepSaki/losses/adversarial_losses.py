import tensorflow as tf
from typing import Iterable

from DeepSaki.losses.loss_helper import manually_reduce_loss


# TODO: generalize for multiple inputs


class AdversarialLossGenerator(tf.keras.losses.Loss):
    def __init__(
        self,
        adversarial_loss_func: tf.keras.losses.Loss,
        global_batch_size: int,
    ) -> None:
        self.adversarial_loss_func = adversarial_loss_func
        self.global_batch_size = global_batch_size

        # ensure reduction is none, since it will be scaled by the global batch size!
        self.adversarial_loss_func.reduction = tf.keras.losses.Reduction.NONE

    def call(
        self,
        generated: tf.Tensor,
    ) -> tf.Tensor:
        gen_loss = self.adversarial_loss_func(tf.ones_like(generated), generated)
        return manually_reduce_loss(gen_loss, self.global_batch_size)


class AdversarialLossDiscriminator(tf.keras.losses.Loss):
    def __init__(
        self,
        adversarial_loss_func: tf.keras.losses.Loss,
        global_batch_size: int,
    ) -> None:
        self.adversarial_loss_func = adversarial_loss_func
        self.global_batch_size = global_batch_size

        # ensure reduction is none, since it will be scaled by the global batch size!
        self.adversarial_loss_func.reduction = tf.keras.losses.Reduction.NONE

    def call(
        self,
        real: tf.Tensor,
        generated: tf.Tensor,
    ) -> tf.Tensor:
        real_loss = self.adversarial_loss_func(tf.ones_like(real), real)
        generated_loss = self.adversarial_loss_func(tf.zeros_like(generated), generated)
        disc_loss = (real_loss + generated_loss) * 0.5
        return manually_reduce_loss(disc_loss, self.global_batch_size)
