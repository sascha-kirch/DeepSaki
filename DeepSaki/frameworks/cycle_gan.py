import os
from typing import Callable
from typing import List
from typing import Tuple

import tensorflow as tf

from DeepSaki.models.model_helper import print_model_parameter_count
from DeepSaki.losses.image_based_losses import PixelDistanceLoss, StructuralSimilarityLoss
from DeepSaki.losses.adversarial_losses import AdversarialLossDiscriminator, AdversarialLossGenerator


# TODO: add param to controll the update rule of the discriminator and the generator. default is alternating!
# TODO: think of general CycleGAN + VoloGAN?
# TODO: mention the loss function and the lambda terms, etc.
# TODO: create private functions for gradient calculation and weight update.
# TODO: add graph for model to docstring

class CycleGAN(tf.keras.Model):
    """Abstraction of a CycleGAN model.

    A CycleGAN is a generative adversarial network consisting of 4 models: 2 generators and 2 discriminators. It might
    be used for translation task, i.e. a domain translation from a source to a target domain, where no ground truth is
    available.

    Info:
        CycleGAN was introduced in [Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks,
        Zhu et. al., 2017](https://arxiv.org/abs/1703.10593)
    """

    def __init__(
        self,
        gen_s_to_t: tf.keras.Model,
        gen_t_to_s: tf.keras.Model,
        dis_t: tf.keras.Model,
        dis_s: tf.keras.Model,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        """Initializes the `CycleGAN` object.

        Args:
            gen_s_to_t (tf.keras.Model): Generator model to translate a sample from the source-domain
                into the target-domain.
            gen_t_to_s (tf.keras.Model): Generator model to translate a sample from the target-domain
                into the source-domain.
            dis_t (tf.keras.Model): Discriminator model to predict weather a sample is a true or fake
                sample of the target domain.
            dis_s (tf.keras.Model): Discriminator model to predict weather a sample is a true or fake
                sample of the source domain.
            lambda_cycle (float): Weighting factor to control the contribution of the cycle consistence loss term.
                Defaults to 10.0.
            lambda_identity (float): Weighting factor to control the contribution of the identity loss term.
                Defaults to 0.5.
        """
        super(CycleGAN, self).__init__()

        self.name = "CycleGAN"
        self.gen_s_to_t = gen_s_to_t
        self.gen_t_to_s = gen_t_to_s
        self.dis_s = dis_s
        self.dis_t = dis_t
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.strategy = tf.distribute.get_strategy()
        policy = tf.keras.mixed_precision.global_policy()
        self.use_mixed_precission = policy.name in ["mixed_float16", "mixed_bfloat16"]

    def compile(
        self,
        optim_gen_s_to_t: tf.keras.optimizers.Optimizer,
        optim_gen_t_to_s: tf.keras.optimizers.Optimizer,
        optim_dis_s: tf.keras.optimizers.Optimizer,
        optim_dis_t: tf.keras.optimizers.Optimizer,
        loss_fn_gen_adv_loss: AdversarialLossGenerator,
        loss_fn_dis_adv_loss: AdversarialLossDiscriminator,
        loss_fn_cycle_loss: tf.keras.losses.Loss,
        loss_fn_identity_loss: tf.keras.losses.Loss,
    ) -> None:
        """Set the optimizer for each model and the loss functions for each loss term.

        Args:
            optim_gen_s_to_t (tf.keras.optimizers.Optimizer): Optimizer for the source to target
                generator model.
            optim_gen_t_to_s (tf.keras.optimizers.Optimizer): Optimizer for the target to source
                generator model.
            optim_dis_s (tf.keras.optimizers.Optimizer): Optimizer for the source discriminator model.
            optim_dis_t (tf.keras.optimizers.Optimizer): Optimizer for the target discriminator model.
            loss_fn_gen_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the generator loss.
            loss_fn_dis_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the discriminator loss.
            loss_fn_cycle_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the cycle consistency loss.
            loss_fn_identity_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the identity loss.
        """
        super(CycleGAN, self).compile()
        self.optim_gen_s_to_t = optim_gen_s_to_t
        self.optim_gen_t_to_s = optim_gen_t_to_s
        self.optim_dis_s = optim_dis_s
        self.optim_dis_t = optim_dis_t
        self.loss_fn_gen_loss = loss_fn_gen_adv_loss
        self.loss_fn_dis_loss = loss_fn_dis_adv_loss
        self.loss_fn_cycle_loss = loss_fn_cycle_loss
        self.loss_fn_identity_loss = loss_fn_identity_loss

    def call(self, batch: tf.Tensor, training:bool = True) -> Tuple[tf.Tensor, ...]:
        input_s, input_t = batch
        generated_t = self.gen_s_to_t(input_s, training=training)
        generated_s = self.gen_t_to_s(input_t, training=training)
        cycled_t = self.gen_t_to_s(generated_t, training=training)
        cycled_s = self.gen_s_to_t(generated_s, training=training)
        identity_t = self.gen_s_to_t(input_t, training=training)
        identity_s = self.gen_t_to_s(input_s, training=training)
        discriminated_generated_t = self.dis_t(generated_t, training=training)
        discriminated_generated_s = self.dis_s(generated_s, training=training)
        discriminated_real_t = self.dis_t(input_t, training=training)
        discriminated_real_s = self.dis_s(input_s, training=training)
        return generated_t, generated_s, cycled_t, cycled_s, identity_t, identity_s, discriminated_real_t, discriminated_real_s,discriminated_generated_t,discriminated_generated_s


    def print_short_summary(self) -> str:
        """Print the information of all models.

        Returns:
            String object that has been printed.
        """
        summary = "--------------------------------------------------\n"
        summary += f"---------------- Summary {self.modelType} ---------------\n"
        summary += "--------------------------------------------------\n"
        summary += print_model_parameter_count(self.gen_s_to_t)
        summary += print_model_parameter_count(self.gen_t_to_s)
        summary += print_model_parameter_count(self.dis_t)
        summary += print_model_parameter_count(self.dis_s)

        print(summary)

        return summary

    def train_step(self, train_data_batch: tf.Tensor) -> List[tf.Tensor]:
        r"""Performs a single optimization step of the optimizer for a single batch.

        Args:
            train_data_batch (tf.Tensor): Single batch of training data. Expected shape [2,] where train_data_batch[0]
                corresponds to the real data from the source domain of shape [batch, ...] and train_data_batch[1]
                corresponds to the real data from the target domain of shape [batch, ...].

        Returns:
            loss: Returns list containing individual loss terms: <br>
                \[0\]: total_gen_loss_t (tf.Tensor) <br>
                \[1\]: total_gen_loss_s (tf.Tensor) <br>
                \[2\]: dis_t_loss (tf.Tensor) <br>
                \[3\]: dis_s_loss (tf.Tensor) <br>
                \[4\]: gen_s_to_t_loss (tf.Tensor) <br>
                \[5\]: gen_t_to_s_loss (tf.Tensor) <br>
                \[6\]: cycle_loss_t (tf.Tensor) <br>
                \[7\]: cycle_loss_s (tf.Tensor) <br>
                \[8\]: identity_loss_t (tf.Tensor) <br>
                \[9\]: identity_loss_s (tf.Tensor) <br>
        """
        real_s, real_t = train_data_batch

        with tf.GradientTape(persistent=True) as tape:
            # Generating fakes
            generated_t = self.gen_s_to_t(real_s, training=True)
            generated_s = self.gen_t_to_s(real_t, training=True)

            # Cycle mapping
            cycled_s = self.gen_t_to_s(generated_t, training=True)
            cycled_t = self.gen_s_to_t(generated_s, training=True)

            # Identity mapping
            identity_s = self.gen_t_to_s(real_s, training=True)
            identity_t = self.gen_s_to_t(real_t, training=True)

            # Discriminator output
            disc_real_s = self.dis_s(real_s, training=True)
            disc_generated_s = self.dis_s(generated_s, training=True)

            disc_real_t = self.dis_t(real_t, training=True)
            disc_generated_t = self.dis_t(generated_t, training=True)

            # Discriminator loss
            dis_s_loss = self.loss_fn_dis_loss(disc_real_s, disc_generated_s)
            dis_t_loss = self.loss_fn_dis_loss(disc_real_t, disc_generated_t)

            # Generator adverserial loss
            gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t)
            gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s)

            # Generator cycle loss
            cycle_loss_t = self.loss_fn_cycle_loss(real_t, cycled_t) * self.lambda_cycle
            cycle_loss_s = self.loss_fn_cycle_loss(real_s, cycled_s) * self.lambda_cycle

            # Generator identity loss
            identity_loss_t = self.loss_fn_identity_loss(real_t, identity_t) * self.lambda_identity
            identity_loss_s = self.loss_fn_identity_loss(real_s, identity_s) * self.lambda_identity

            # Total generator loss
            total_gen_loss_t = gen_s_to_t_loss + cycle_loss_t + identity_loss_t
            total_gen_loss_s = gen_t_to_s_loss + cycle_loss_s + identity_loss_s

            if self.use_mixed_precission:
                total_gen_loss_t = self.optim_gen_s_to_t.get_scaled_loss(total_gen_loss_t)
                total_gen_loss_s = self.optim_gen_t_to_s.get_scaled_loss(total_gen_loss_s)
                dis_s_loss = self.optim_dis_s.get_scaled_loss(dis_s_loss)
                dis_t_loss = self.optim_dis_t.get_scaled_loss(dis_t_loss)

        # Get the gradients
        if self.use_mixed_precission:
            scaled_grads_gen_t = tape.gradient(total_gen_loss_t, self.gen_s_to_t.trainable_variables)
            scaled_grads_gen_s = tape.gradient(total_gen_loss_s, self.gen_t_to_s.trainable_variables)
            scaled_grads_dis_s = tape.gradient(dis_s_loss, self.dis_s.trainable_variables)
            scaled_grads_dis_t = tape.gradient(dis_t_loss, self.dis_t.trainable_variables)

            grads_gen_s_to_t = self.optim_gen_s_to_t.get_unscaled_gradients(scaled_grads_gen_t)
            grads_gen_t_to_s = self.optim_gen_t_to_s.get_unscaled_gradients(scaled_grads_gen_s)
            grads_dis_s = self.optim_dis_s.get_unscaled_gradients(scaled_grads_dis_s)
            grads_dis_t = self.optim_dis_t.get_unscaled_gradients(scaled_grads_dis_t)
        else:
            grads_gen_s_to_t = tape.gradient(total_gen_loss_t, self.gen_s_to_t.trainable_variables)
            grads_gen_t_to_s = tape.gradient(total_gen_loss_s, self.gen_t_to_s.trainable_variables)
            grads_dis_s = tape.gradient(dis_s_loss, self.dis_s.trainable_variables)
            grads_dis_t = tape.gradient(dis_t_loss, self.dis_t.trainable_variables)

        # Update the weights
        self.optim_gen_s_to_t.apply_gradients(zip(grads_gen_s_to_t, self.gen_s_to_t.trainable_variables))
        self.optim_gen_t_to_s.apply_gradients(zip(grads_gen_t_to_s, self.gen_t_to_s.trainable_variables))

        self.optim_dis_s.apply_gradients(zip(grads_dis_s, self.dis_s.trainable_variables))
        self.optim_dis_t.apply_gradients(zip(grads_dis_t, self.dis_t.trainable_variables))

        return [
            total_gen_loss_t,
            total_gen_loss_s,
            dis_t_loss,
            dis_s_loss,
            gen_s_to_t_loss,
            gen_t_to_s_loss,
            cycle_loss_t,
            cycle_loss_s,
            identity_loss_t,
            identity_loss_s,
        ]

    def test_step(self, test_data_batch: tf.Tensor) -> List[tf.Tensor]:
        r"""Performs a single test step without updating the weights for a single batch.

        Args:
            test_data_batch (tf.Tensor): Single batch of test data. Expected shape [2,] where test_data_batch[0]
                corresponds to the real data from the source domain of shape [batch, ...] and test_data_batch[1]
                corresponds to the real data from the target domain of shape [batch, ...].
                Further, it is expected, that the data from the source domain can be passed directly into the model. In
                case the model expects multiple inputs, the model must unpack them itself.

        Returns:
            loss: Returns list containing individual loss terms: <br>
                \[0\]: total_gen_loss_t (tf.Tensor) <br>
                \[1\]: total_gen_loss_s (tf.Tensor) <br>
                \[2\]: dis_t_loss (tf.Tensor) <br>
                \[3\]: dis_s_loss (tf.Tensor) <br>
                \[4\]: gen_s_to_t_loss (tf.Tensor) <br>
                \[5\]: gen_t_to_s_loss (tf.Tensor) <br>
                \[6\]: cycle_loss_t (tf.Tensor) <br>
                \[7\]: cycle_loss_s (tf.Tensor) <br>
                \[8\]: identity_loss_t (tf.Tensor) <br>
                \[9\]: identity_loss_s (tf.Tensor) <br>
        """
        real_s, real_t = test_data_batch

        # Generate fakes
        generated_t = self.gen_s_to_t(real_s, training=False)
        generated_s = self.gen_t_to_s(real_t, training=False)

        # Cycle mapping
        cycled_s = self.gen_t_to_s(generated_t, training=False)
        cycled_t = self.gen_s_to_t(generated_s, training=False)

        # Identity mapping
        same_s = self.gen_t_to_s(real_s, training=False)
        same_t = self.gen_s_to_t(real_t, training=False)

        # Discriminator output
        disc_real_s = self.dis_s(real_s, training=False)
        disc_generated_s = self.dis_s(generated_s, training=False)

        disc_real_t = self.dis_t(real_t, training=False)
        disc_generated_t = self.dis_t(generated_t, training=False)

        # Discriminator loss
        dis_s_loss = self.loss_fn_dis_loss(disc_real_s, disc_generated_s)
        dis_t_loss = self.loss_fn_dis_loss(disc_real_t, disc_generated_t)

        # Generator adverserial loss
        gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t)
        gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s)

        # Generator cycle loss
        cycle_loss_t = self.loss_fn_cycle_loss(real_t, cycled_t) * self.lambda_cycle
        cycle_loss_s = self.loss_fn_cycle_loss(real_s, cycled_s) * self.lambda_cycle

        # Generator identity loss
        identity_loss_t = self.loss_fn_identity_loss(real_t, same_t) * self.lambda_identity
        identity_loss_s = self.loss_fn_identity_loss(real_s, same_s) * self.lambda_identity

        # Total generator loss
        total_gen_loss_t = gen_s_to_t_loss + cycle_loss_t + identity_loss_t
        total_gen_loss_s = gen_t_to_s_loss + cycle_loss_s + identity_loss_s

        return [
            total_gen_loss_t,
            total_gen_loss_s,
            dis_t_loss,
            dis_s_loss,
            gen_s_to_t_loss,
            gen_t_to_s_loss,
            cycle_loss_t,
            cycle_loss_s,
            identity_loss_t,
            identity_loss_s,
        ]

    @tf.function
    def distributed_train_step(self, train_data_batch: tf.distribute.DistributedValues) -> List[tf.Tensor]:
        """Distributes a single training step to all workers, collects the result and accumulates the individual losses.

        Function is compiled with the `tf.function` decorator.

        Args:
            train_data_batch (tf.distribute.DistributedValues): Object containing training data of a single batch, one
                for each replica it will be distributed to. Can be obtained when iterating over a distributed dataset
                `tf.distribute.DistributedDataset`.

        Returns:
            List of reduced loss values from all replicas. Shape and definition same as in `CycleGAN.train_step(...)`.
        """
        per_replica_loss_vector = self.strategy.run(self.train_step, args=(train_data_batch,))

        # reduce the result of the replicas for every loss value returned!
        return [
            self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            for per_replica_loss in per_replica_loss_vector
        ]

    @tf.function
    def distributed_test_step(self, test_data_batch: tf.distribute.DistributedValues) -> List[tf.Tensor]:
        """Distributes a single test step to all workers, collects the result and accumulates the individual losses.

        Function is compiled with the `tf.function` decorator.

        Args:
            test_data_batch (tf.distribute.DistributedValues): Object containing testing data of a single batch, one
                for each replica it will be distributed to. Can be obtained when iterating over a distributed dataset
                `tf.distribute.DistributedDataset`.

        Returns:
            List of reduced loss values from all replicas. Shape and definition same as in `CycleGAN.test_step(...)`.
        """
        per_replica_loss_vector = self.strategy.run(self.test_step, args=(test_data_batch,))

        # reduce the result of the replicas for every loss value returned!
        return [
            self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)
            for per_replica_loss in per_replica_loss_vector
        ]

    def save_models(self, base_directory: str) -> None:
        """Save both generators and both discriminator models.

        Args:
            base_directory (str): Directory where the models shall be saved. For each model, one directory is creacted.
        """
        save_options = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        self.gen_t_to_s.save(os.path.join(base_directory, "gen_t_to_s"), options=save_options)
        self.gen_s_to_t.save(os.path.join(base_directory, "gen_s_to_t"), options=save_options)
        self.dis_s.save(os.path.join(base_directory, "dis_s"), options=save_options)
        self.dis_t.save(os.path.join(base_directory, "dis_t"), options=save_options)

    def load_models(
        self,
        path_gen_s_to_t: str,
        path_gen_t_to_s: str,
        path_dis_t: str,
        path_dis_s: str,
        print_summary: bool = False,
    ) -> None:
        """Load weights for each of the models.

        Args:
            path_gen_s_to_t (str): Path to the weights of the source-to-target generator.
            path_gen_t_to_s (str): Path to the weights of the target-to-source generator.
            path_dis_t (str): Path to the weights of the target discriminator.
            path_dis_s (str): Path to the weights of the source discriminator.
            print_summary (bool, optional): Weathor or not to print the sumarry of the loaded models. Defaults to False.
        """
        load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
        self.gen_s_to_t = tf.keras.models.load_model(path_gen_s_to_t, options=load_options)
        self.gen_t_to_s = tf.keras.models.load_model(path_gen_t_to_s, options=load_options)
        self.dis_t = tf.keras.models.load_model(path_dis_t, options=load_options)
        self.dis_s = tf.keras.models.load_model(path_dis_s, options=load_options)

        if print_summary:
            self.print_short_summary()


# TODO: add consistency loss and consistency lambda that are used as a loss function when applying cut_mix or cut_out.
class VoloGAN(CycleGAN):
    def __init__(
        self,
        gen_s_to_t: tf.keras.Model,
        gen_t_to_s: tf.keras.Model,
        dis_t: tf.keras.Model,
        dis_s: tf.keras.Model,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        lambda_ssim: float = 1.0,
        use_cutmix: bool = False,
        cutmix_probability: float = 0.5,
    ) -> None:
        super(VoloGAN, self).__init__(
            gen_s_to_t,
            gen_t_to_s,
            dis_s,
            dis_t,
            lambda_cycle,
            lambda_identity,
        )
        self.name = "VoloGAN"
        self.lambda_ssim = lambda_ssim
        self.useCutmix = use_cutmix
        self.cutmixProbability = cutmix_probability

    # To reduce python overhead, and maximize the performance of your TPU, try out the experimental experimental_steps_per_execution argument to Model.compile. Here it increases throughput by about 50%:
    # https://www.tensorflow.org/guide/tpu#train_a_model_using_keras_high_level_apis

    def compile(
        self,
        optim_gen_s_to_t: tf.keras.optimizers.Optimizer,
        optim_gen_t_to_s: tf.keras.optimizers.Optimizer,
        optim_dis_s: tf.keras.optimizers.Optimizer,
        optim_dis_t: tf.keras.optimizers.Optimizer,
        loss_fn_gen_adv_loss: AdversarialLossGenerator,
        loss_fn_dis_adv_loss: AdversarialLossDiscriminator,
        loss_fn_cycle_loss: PixelDistanceLoss,
        loss_fn_identity_loss: PixelDistanceLoss,
        ssim_loss_fn: StructuralSimilarityLoss,
    ) -> None:
        super(VoloGAN, self).compile(
            optim_gen_s_to_t,
            optim_gen_t_to_s,
            optim_dis_s,
            optim_dis_t,
            loss_fn_gen_adv_loss,
            loss_fn_dis_adv_loss,
            loss_fn_cycle_loss,
            loss_fn_identity_loss,
        )
        self.ssim_loss_fn = ssim_loss_fn

    def train_step(self, train_data_batch: tf.Tensor) -> List[tf.Tensor]:
        real_s, real_t = train_data_batch

        with tf.GradientTape(persistent=True) as tape:
            # Generating fakes
            generated_t = self.gen_s_to_t(real_s, training=True)
            generated_s = self.gen_t_to_s(real_t, training=True)

            # Cycle mapping
            cycled_s = self.gen_t_to_s(generated_t, training=True)
            cycled_t = self.gen_s_to_t(generated_s, training=True)

            # Identity mapping
            identity_s = self.gen_t_to_s(real_s, training=True)
            identity_t = self.gen_s_to_t(real_t, training=True)

            if DISCRIMINATOR_DESIGN == "PatchGAN":
                # Discriminator output
                disc_real_s = self.dis_s(real_s, training=True)
                disc_generated_s = self.dis_s(generated_s, training=True)

                disc_real_t = self.dis_t(real_t, training=True)
                disc_generated_t = self.dis_t(generated_t, training=True)

                # Discriminator loss
                dis_s_loss = self.loss_fn_dis_loss(disc_real_s, disc_generated_s)
                dis_t_loss = self.loss_fn_dis_loss(disc_real_t, disc_generated_t)

                # Generator adverserial loss
                gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t)
                gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s)

            elif DISCRIMINATOR_DESIGN == "UNet":
                # Discriminator output
                disc_real_s_encoder, disc_real_s_decoder = self.dis_s(real_s, training=True)
                disc_generated_s_encoder, disc_generated_s_decoder = self.dis_s(generated_s, training=True)
                disc_real_t_encoder, disc_real_t_decoder = self.dis_t(real_t, training=True)
                disc_generated_t_encoder, disc_generated_t_decoder = self.dis_t(generated_t, training=True)

                if self.useCutmix and np.random.uniform(0, 1) < self.cutmixProbability:
                    batch_size, height, width, channel = rgbd_synthetic.shape
                    mixing_mask = GetCutmixMask(batch_size, height, width, channel)
                    _, cutmix_s = CutMix(real_s, generated_s, ignoreBackground=True, mixing_mask=mixing_mask)
                    _, cutmix_t = CutMix(real_t, generated_t, ignoreBackground=True, mixing_mask=mixing_mask)

                    # calculate discriminator output of cutmix images. Only decoder output relevant!
                    _, disc_cutmix_s = self.dis_s(cutmix_s, training=True)
                    _, disc_cutmix_t = self.dis_t(cutmix_t, training=True)

                    # calculate cutmix of discriminator output
                    _, cutmix_disc_s = CutMix(
                        disc_real_s_decoder,
                        disc_generated_s_decoder,
                        ignoreBackground=True,
                        mixing_mask=mixing_mask,
                    )
                    _, cutmix_disc_t = CutMix(
                        disc_real_t_decoder,
                        disc_generated_t_decoder,
                        ignoreBackground=True,
                        mixing_mask=mixing_mask,
                    )

                else:
                    disc_cutmix_s = None
                    disc_cutmix_t = None
                    cutmix_disc_s = None
                    cutmix_disc_t = None

                # Discriminator loss
                dis_s_loss = self.loss_fn_dis_loss(
                    disc_real_s_encoder,
                    disc_real_s_decoder,
                    disc_generated_s_encoder,
                    disc_generated_s_decoder,
                    disc_cutmix_s,
                    cutmix_disc_s,
                )
                dis_t_loss = self.loss_fn_dis_loss(
                    disc_real_t_encoder,
                    disc_real_t_decoder,
                    disc_generated_t_encoder,
                    disc_generated_t_decoder,
                    disc_cutmix_t,
                    cutmix_disc_t,
                )

                # Generator adverserial loss
                gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t_encoder, disc_generated_t_decoder)
                gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s_encoder, disc_generated_s_decoder)

            elif DISCRIMINATOR_DESIGN == "OneShot_GAN":
                # Discriminator output
                (
                    disc_real_s_low_level,
                    disc_real_s_layout,
                    disc_real_s_content,
                ) = self.dis_s(real_s, training=True)
                (
                    disc_generated_s_low_level,
                    disc_generated_s_layout,
                    disc_generated_s_content,
                ) = self.dis_s(generated_s, training=True)

                (
                    disc_real_t_low_level,
                    disc_real_t_layout,
                    disc_real_t_content,
                ) = self.dis_t(real_t, training=True)
                (
                    disc_generated_t_low_level,
                    disc_generated_t_layout,
                    disc_generated_t_content,
                ) = self.dis_t(generated_t, training=True)

                # Discriminator loss
                dis_s_loss = self.loss_fn_dis_loss(
                    disc_real_s_low_level,
                    disc_real_s_layout,
                    disc_real_s_content,
                    disc_generated_s_low_level,
                    disc_generated_s_layout,
                    disc_generated_s_content,
                )
                dis_t_loss = self.loss_fn_dis_loss(
                    disc_real_t_low_level,
                    disc_real_t_layout,
                    disc_real_t_content,
                    disc_generated_t_low_level,
                    disc_generated_t_layout,
                    disc_generated_t_content,
                )

                # Generator adverserial loss
                gen_s_to_t_loss = self.loss_fn_gen_loss(
                    disc_generated_t_low_level, disc_generated_t_layout, disc_generated_t_content
                )
                gen_t_to_s_loss = self.loss_fn_gen_loss(
                    disc_generated_s_low_level, disc_generated_s_layout, disc_generated_s_content
                )

            else:
                raise Exception("Discriminator is not Defined")

            # Generator cycle loss
            cycle_loss_t = self.loss_fn_cycle_loss(real_t, cycled_t) * self.lambda_cycle
            cycle_loss_s = self.loss_fn_cycle_loss(real_s, cycled_s) * self.lambda_cycle

            # Generator identity loss
            identity_loss_t = self.loss_fn_identity_loss(real_t, identity_t) * self.lambda_identity
            identity_loss_s = self.loss_fn_identity_loss(real_s, identity_s) * self.lambda_identity

            # Generator SSIM Loss
            ssim_loss_t = self.ssim_loss_fn(real_t, cycled_t) * self.lambda_ssim
            ssim_loss_s = self.ssim_loss_fn(real_s, cycled_s) * self.lambda_ssim

            # Total generator loss
            total_gen_loss_t = gen_s_to_t_loss + cycle_loss_t + identity_loss_t + ssim_loss_t
            total_gen_loss_s = gen_t_to_s_loss + cycle_loss_s + identity_loss_s + ssim_loss_s

        # Get the gradients for the generators
        grads_t = tape.gradient(total_gen_loss_t, self.gen_s_to_t.trainable_variables)
        grads_s = tape.gradient(total_gen_loss_s, self.gen_t_to_s.trainable_variables)

        # Get the gradients for the discriminators
        dis_s_grads = tape.gradient(dis_s_loss, self.dis_s.trainable_variables)
        dis_t_grads = tape.gradient(dis_t_loss, self.dis_t.trainable_variables)

        # Update the weights of the generators
        self.optim_gen_s_to_t.apply_gradients(zip(grads_t, self.gen_s_to_t.trainable_variables))
        self.optim_gen_t_to_s.apply_gradients(zip(grads_s, self.gen_t_to_s.trainable_variables))

        # Update the weights of the discriminators
        self.optim_dis_s.apply_gradients(zip(dis_s_grads, self.dis_s.trainable_variables))
        self.optim_dis_t.apply_gradients(zip(dis_t_grads, self.dis_t.trainable_variables))
        return [
            total_gen_loss_t,
            total_gen_loss_s,
            dis_t_loss,
            dis_s_loss,
            gen_s_to_t_loss,
            gen_t_to_s_loss,
            cycle_loss_t,
            cycle_loss_s,
            identity_loss_t,
            identity_loss_s,
            ssim_loss_t,
            ssim_loss_s,
        ]

    def test_step(self, test_data_batch: tf.Tensor) -> List[tf.Tensor]:
        real_s, real_t = test_data_batch

        # Generate fakes
        generated_t = self.gen_s_to_t(real_s, training=False)
        generated_s = self.gen_t_to_s(real_t, training=False)

        # Cycle mapping
        cycled_s = self.gen_t_to_s(generated_t, training=False)
        cycled_t = self.gen_s_to_t(generated_s, training=False)

        # Identity mapping
        same_s = self.gen_t_to_s(real_s, training=False)
        same_t = self.gen_s_to_t(real_t, training=False)

        if DISCRIMINATOR_DESIGN == "PatchGAN":
            # Discriminator output
            disc_real_s = self.dis_s(real_s, training=False)
            disc_generated_s = self.dis_s(generated_s, training=False)

            disc_real_t = self.dis_t(real_t, training=False)
            disc_generated_t = self.dis_t(generated_t, training=False)

            # Discriminator loss
            dis_s_loss = self.loss_fn_dis_loss(disc_real_s, disc_generated_s)
            dis_t_loss = self.loss_fn_dis_loss(disc_real_t, disc_generated_t)

            # Generator adverserial loss
            gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t)
            gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s)

        elif DISCRIMINATOR_DESIGN == "UNet":
            # Discriminator output
            disc_real_s_encoder, disc_real_s_decoder = self.dis_s(real_s, training=False)
            disc_generated_s_encoder, disc_generated_s_decoder = self.dis_s(generated_s, training=False)
            disc_real_t_encoder, disc_real_t_decoder = self.dis_t(real_t, training=False)
            disc_generated_t_encoder, disc_generated_t_decoder = self.dis_t(generated_t, training=False)

            # Discriminator loss -> No Cutmix Regularization during Test!
            dis_s_loss = self.loss_fn_dis_loss(
                disc_real_s_encoder,
                disc_real_s_decoder,
                disc_generated_s_encoder,
                disc_generated_s_decoder,
                None,
                None,
            )
            dis_t_loss = self.loss_fn_dis_loss(
                disc_real_t_encoder,
                disc_real_t_decoder,
                disc_generated_t_encoder,
                disc_generated_t_decoder,
                None,
                None,
            )

            # Generator adverserial loss
            gen_s_to_t_loss = self.loss_fn_gen_loss(disc_generated_t_encoder, disc_generated_t_decoder)
            gen_t_to_s_loss = self.loss_fn_gen_loss(disc_generated_s_encoder, disc_generated_s_decoder)

        elif DISCRIMINATOR_DESIGN == "OneShot_GAN":
            # Discriminator output

            disc_real_s_low_level, disc_real_s_layout, disc_real_s_content = self.dis_s(real_s, training=False)
            (
                disc_generated_s_low_level,
                disc_generated_s_layout,
                disc_generated_s_content,
            ) = self.dis_s(generated_s, training=False)

            disc_real_t_low_level, disc_real_t_layout, disc_real_t_content = self.dis_t(real_t, training=False)
            (
                disc_generated_t_low_level,
                disc_generated_t_layout,
                disc_generated_t_content,
            ) = self.dis_t(generated_t, training=False)

            # Discriminator loss
            dis_s_loss = self.loss_fn_dis_loss(
                disc_real_s_low_level,
                disc_real_s_layout,
                disc_real_s_content,
                disc_generated_s_low_level,
                disc_generated_s_layout,
                disc_generated_s_content,
            )
            dis_t_loss = self.loss_fn_dis_loss(
                disc_real_t_low_level,
                disc_real_t_layout,
                disc_real_t_content,
                disc_generated_t_low_level,
                disc_generated_t_layout,
                disc_generated_t_content,
            )

            # Generator adverserial loss
            gen_s_to_t_loss = self.loss_fn_gen_loss(
                disc_generated_t_low_level, disc_generated_t_layout, disc_generated_t_content
            )
            gen_t_to_s_loss = self.loss_fn_gen_loss(
                disc_generated_s_low_level, disc_generated_s_layout, disc_generated_s_content
            )

        else:
            raise Exception("Discriminator is not Defined")

        # Generator cycle loss
        cycle_loss_t = self.loss_fn_cycle_loss(real_t, cycled_t) * self.lambda_cycle
        cycle_loss_s = self.loss_fn_cycle_loss(real_s, cycled_s) * self.lambda_cycle

        # Generator identity loss
        identity_loss_t = self.loss_fn_identity_loss(real_t, same_t) * self.lambda_identity
        identity_loss_s = self.loss_fn_identity_loss(real_s, same_s) * self.lambda_identity

        # Generator SSIM Loss
        ssim_loss_t = self.ssim_loss_fn(real_t, cycled_t) * self.lambda_ssim
        ssim_loss_s = self.ssim_loss_fn(real_s, cycled_s) * self.lambda_ssim

        # Total generator loss
        total_gen_loss_t = gen_s_to_t_loss + cycle_loss_t + identity_loss_t + ssim_loss_t
        total_gen_loss_s = gen_t_to_s_loss + cycle_loss_s + identity_loss_s + ssim_loss_s

        return [
            total_gen_loss_t,
            total_gen_loss_s,
            dis_t_loss,
            dis_s_loss,
            gen_s_to_t_loss,
            gen_t_to_s_loss,
            cycle_loss_t,
            cycle_loss_s,
            identity_loss_t,
            identity_loss_s,
            ssim_loss_t,
            ssim_loss_s,
        ]
