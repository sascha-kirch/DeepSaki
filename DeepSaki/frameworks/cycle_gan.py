import os
from typing import Callable
from typing import List
from typing import Tuple

import tensorflow as tf

from DeepSaki.models.model_helper import print_model_parameter_count


# TODO: add param to controll the update rule of the discriminator and the generator. default is alternating!
# TODO: think of general CycleGAN + VoloGAN?
# TODO: mention the loss function and the lambda terms, etc.


# TODO: add graph for model
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
        generator_source_to_target: tf.keras.Model,
        generator_target_to_source: tf.keras.Model,
        discriminator_target: tf.keras.Model,
        discriminator_source: tf.keras.Model,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
    ) -> None:
        """Initializes the `CycleGAN` object.

        Args:
            generator_source_to_target (tf.keras.Model): Generator model to translate a sample from the source-domain
                into the target-domain.
            generator_target_to_source (tf.keras.Model): Generator model to translate a sample from the target-domain
                into the source-domain.
            discriminator_target (tf.keras.Model): Discriminator model to predict weather a sample is a true or fake
                sample of the target domain.
            discriminator_source (tf.keras.Model): Discriminator model to predict weather a sample is a true or fake
                sample of the source domain.
            lambda_cycle (float): Weighting factor to control the contribution of the cycle consistence loss term.
                Defaults to 10.0.
            lambda_identity (float): Weighting factor to control the contribution of the identity loss term.
                Defaults to 0.5.
        """
        super(CycleGAN, self).__init__()

        self.modelType = "CycleGAN"  # TODO: check how this was used und change to model name and pass to base class
        self.generator_source_to_target = generator_source_to_target
        self.generator_target_to_source = generator_target_to_source
        self.discriminator_source = discriminator_source
        self.discriminator_target = discriminator_target
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.strategy = tf.distribute.get_strategy()

    def compile(
        self,
        optimizer_generator_source_to_target: tf.keras.optimizers.Optimizer,
        optimizer_generator_target_to_source: tf.keras.optimizers.Optimizer,
        optimizer_discriminator_source: tf.keras.optimizers.Optimizer,
        optimizer_discriminator_target: tf.keras.optimizers.Optimizer,
        loss_fn_generator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_discriminator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_cycle_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_identity_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> None:
        """Set the optimizer for each model and the loss functions for each loss term.

        Args:
            optimizer_generator_source_to_target (tf.keras.optimizers.Optimizer): Optimizer for the source to target
                generator model.
            optimizer_generator_target_to_source (tf.keras.optimizers.Optimizer): Optimizer for the target to source
                generator model.
            optimizer_discriminator_source (tf.keras.optimizers.Optimizer): Optimizer for the source discriminator model.
            optimizer_discriminator_target (tf.keras.optimizers.Optimizer): Optimizer for the target discriminator model.
            loss_fn_generator_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the generator loss.
            loss_fn_discriminator_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the discriminator loss.
            loss_fn_cycle_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the cycle consistency loss.
            loss_fn_identity_loss (Callable[[tf.Tensor, tf.Tensor], tf.Tensor]): Loss function for the identity loss.
        """
        super(CycleGAN, self).compile()
        self.optimizer_generator_source_to_target = optimizer_generator_source_to_target
        self.optimizer_generator_target_to_source = optimizer_generator_target_to_source
        self.optimizer_discriminator_source = optimizer_discriminator_source
        self.optimizer_discriminator_target = optimizer_discriminator_target
        self.loss_fn_generator_loss = loss_fn_generator_loss
        self.loss_fn_discriminator_loss = loss_fn_discriminator_loss
        self.loss_fn_cycle_loss = loss_fn_cycle_loss
        self.loss_fn_identity_loss = loss_fn_identity_loss

    # TODO: check if makes sense? Should the discriminator also infere the generator output?
    def call(self, batch: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        input_source, input_target = batch
        generated_target = self.generator_source_to_target(input_source)
        generated_source = self.generator_target_to_source(input_target)
        discriminated_target = self.discriminator_target(input_target)
        discriminated_source = self.discriminator_source(input_source)
        return generated_target, generated_source, discriminated_target, discriminated_source

    # TODO: simplify method
    def print_short_summary(self) -> str:
        """Print the information of all models.

        Returns:
            String object that has been printed.
        """
        summary = "--------------------------------------------------\n"
        summary += f"---------------- Summary {self.modelType} ---------------\n"
        summary += "--------------------------------------------------\n"
        summary += print_model_parameter_count(self.generator_source_to_target)
        summary += print_model_parameter_count(self.generator_target_to_source)
        summary += print_model_parameter_count(self.discriminator_target)
        summary += print_model_parameter_count(self.discriminator_source)

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
                \[0\]: total_gen_loss_target (tf.Tensor) <br>
                \[1\]: total_gen_loss_source (tf.Tensor) <br>
                \[2\]: discriminator_target_loss (tf.Tensor) <br>
                \[3\]: discriminator_source_loss (tf.Tensor) <br>
                \[4\]: generator_source_to_target_loss (tf.Tensor) <br>
                \[5\]: generator_target_to_source_loss (tf.Tensor) <br>
                \[6\]: cycle_loss_target (tf.Tensor) <br>
                \[7\]: cycle_loss_source (tf.Tensor) <br>
                \[8\]: identity_loss_target (tf.Tensor) <br>
                \[9\]: identity_loss_source (tf.Tensor) <br>
        """
        real_source, real_target = train_data_batch

        with tf.GradientTape(persistent=True) as tape:
            # Generating fakes
            generated_target = self.generator_source_to_target(real_source, training=True)
            generated_source = self.generator_target_to_source(real_target, training=True)

            # Cycle mapping
            cycled_source = self.generator_target_to_source(generated_target, training=True)
            cycled_target = self.generator_source_to_target(generated_source, training=True)

            # Identity mapping
            # TODO: change naming
            same_source = self.generator_target_to_source(real_source, training=True)
            same_target = self.generator_source_to_target(real_target, training=True)

            # Discriminator output
            disc_real_source = self.discriminator_source(real_source, training=True)
            disc_generated_source = self.discriminator_source(generated_source, training=True)

            disc_real_target = self.discriminator_target(real_target, training=True)
            disc_generated_target = self.discriminator_target(generated_target, training=True)

            # Discriminator loss
            discriminator_source_loss = self.loss_fn_discriminator_loss(disc_real_source, disc_generated_source)
            discriminator_target_loss = self.loss_fn_discriminator_loss(disc_real_target, disc_generated_target)

            # Generator adverserial loss
            generator_source_to_target_loss = self.loss_fn_generator_loss(disc_generated_target)
            generator_target_to_source_loss = self.loss_fn_generator_loss(disc_generated_source)

            # Generator cycle loss
            cycle_loss_target = self.loss_fn_cycle_loss(real_target, cycled_target) * self.lambda_cycle
            cycle_loss_source = self.loss_fn_cycle_loss(real_source, cycled_source) * self.lambda_cycle

            # Generator identity loss
            identity_loss_target = self.loss_fn_identity_loss(real_target, same_target) * self.lambda_identity
            identity_loss_source = self.loss_fn_identity_loss(real_source, same_source) * self.lambda_identity

            # Total generator loss
            total_gen_loss_target = generator_source_to_target_loss + cycle_loss_target + identity_loss_target
            total_gen_loss_source = generator_target_to_source_loss + cycle_loss_source + identity_loss_source

        # Get the gradients for the generators
        grads_target = tape.gradient(total_gen_loss_target, self.generator_source_to_target.trainable_variables)
        grads_source = tape.gradient(total_gen_loss_source, self.generator_target_to_source.trainable_variables)

        # Get the gradients for the discriminators
        discriminator_source_grads = tape.gradient(
            discriminator_source_loss, self.discriminator_source.trainable_variables
        )
        discriminator_target_grads = tape.gradient(
            discriminator_target_loss, self.discriminator_target.trainable_variables
        )

        # Update the weights of the generators
        self.optimizer_generator_source_to_target.apply_gradients(
            zip(grads_target, self.generator_source_to_target.trainable_variables)
        )
        self.optimizer_generator_target_to_source.apply_gradients(
            zip(grads_source, self.generator_target_to_source.trainable_variables)
        )

        # Update the weights of the discriminators
        self.optimizer_discriminator_source.apply_gradients(
            zip(discriminator_source_grads, self.discriminator_source.trainable_variables)
        )
        self.optimizer_discriminator_target.apply_gradients(
            zip(discriminator_target_grads, self.discriminator_target.trainable_variables)
        )

        return [
            total_gen_loss_target,
            total_gen_loss_source,
            discriminator_target_loss,
            discriminator_source_loss,
            generator_source_to_target_loss,
            generator_target_to_source_loss,
            cycle_loss_target,
            cycle_loss_source,
            identity_loss_target,
            identity_loss_source,
        ]

    def test_step(self, test_data_batch: tf.Tensor) -> List[tf.Tensor]:
        r"""Performs a single test step without updating the weights for a single batch.

        Args:
            test_data_batch (tf.Tensor): Single batch of test data. Expected shape [2,] where test_data_batch[0]
                corresponds to the real data from the source domain of shape [batch, ...] and test_data_batch[1]
                corresponds to the real data from the target domain of shape [batch, ...].

        Returns:
            loss: Returns list containing individual loss terms: <br>
                \[0\]: total_gen_loss_target (tf.Tensor) <br>
                \[1\]: total_gen_loss_source (tf.Tensor) <br>
                \[2\]: discriminator_target_loss (tf.Tensor) <br>
                \[3\]: discriminator_source_loss (tf.Tensor) <br>
                \[4\]: generator_source_to_target_loss (tf.Tensor) <br>
                \[5\]: generator_target_to_source_loss (tf.Tensor) <br>
                \[6\]: cycle_loss_target (tf.Tensor) <br>
                \[7\]: cycle_loss_source (tf.Tensor) <br>
                \[8\]: identity_loss_target (tf.Tensor) <br>
                \[9\]: identity_loss_source (tf.Tensor) <br>
        """
        real_source, real_target = test_data_batch

        # Generate fakes
        generated_target = self.generator_source_to_target(real_source, training=False)
        generated_source = self.generator_target_to_source(real_target, training=False)

        # Cycle mapping
        cycled_source = self.generator_target_to_source(generated_target, training=False)
        cycled_target = self.generator_source_to_target(generated_source, training=False)

        # Identity mapping
        same_source = self.generator_target_to_source(real_source, training=False)
        same_target = self.generator_source_to_target(real_target, training=False)

        # Discriminator output
        disc_real_source = self.discriminator_source(real_source, training=False)
        disc_generated_source = self.discriminator_source(generated_source, training=False)

        disc_real_target = self.discriminator_target(real_target, training=False)
        disc_generated_target = self.discriminator_target(generated_target, training=False)

        # Discriminator loss
        discriminator_source_loss = self.loss_fn_discriminator_loss(disc_real_source, disc_generated_source)
        discriminator_target_loss = self.loss_fn_discriminator_loss(disc_real_target, disc_generated_target)

        # Generator adverserial loss
        generator_source_to_target_loss = self.loss_fn_generator_loss(disc_generated_target)
        generator_target_to_source_loss = self.loss_fn_generator_loss(disc_generated_source)

        # Generator cycle loss
        cycle_loss_target = self.loss_fn_cycle_loss(real_target, cycled_target) * self.lambda_cycle
        cycle_loss_source = self.loss_fn_cycle_loss(real_source, cycled_source) * self.lambda_cycle

        # Generator identity loss
        identity_loss_target = self.loss_fn_identity_loss(real_target, same_target) * self.lambda_identity
        identity_loss_source = self.loss_fn_identity_loss(real_source, same_source) * self.lambda_identity

        # Total generator loss
        total_gen_loss_target = generator_source_to_target_loss + cycle_loss_target + identity_loss_target
        total_gen_loss_source = generator_target_to_source_loss + cycle_loss_source + identity_loss_source

        return [
            total_gen_loss_target,
            total_gen_loss_source,
            discriminator_target_loss,
            discriminator_source_loss,
            generator_source_to_target_loss,
            generator_target_to_source_loss,
            cycle_loss_target,
            cycle_loss_source,
            identity_loss_target,
            identity_loss_source,
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
        self.generator_target_to_source.save(
            os.path.join(base_directory, "generator_target_to_source"), options=save_options
        )
        self.generator_source_to_target.save(
            os.path.join(base_directory, "generator_source_to_target"), options=save_options
        )
        self.discriminator_source.save(os.path.join(base_directory, "discriminator_source"), options=save_options)
        self.discriminator_target.save(os.path.join(base_directory, "discriminator_target"), options=save_options)

    def load_models(
        self,
        path_generator_source_to_target: str,
        path_generator_target_to_source: str,
        path_discriminator_target: str,
        path_discriminator_source: str,
        print_summary: bool = False,
    ) -> None:
        """Load weights for each of the models.

        Args:
            path_generator_source_to_target (str): Path to the weights of the source-to-target generator.
            path_generator_target_to_source (str): Path to the weights of the target-to-source generator.
            path_discriminator_target (str): Path to the weights of the target discriminator.
            path_discriminator_source (str): Path to the weights of the source discriminator.
            print_summary (bool, optional): Weathor or not to print the sumarry of the loaded models. Defaults to False.
        """
        load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")
        self.generator_source_to_target = tf.keras.models.load_model(
            path_generator_source_to_target, options=load_options
        )
        self.generator_target_to_source = tf.keras.models.load_model(
            path_generator_target_to_source, options=load_options
        )
        self.discriminator_target = tf.keras.models.load_model(path_discriminator_target, options=load_options)
        self.discriminator_source = tf.keras.models.load_model(path_discriminator_source, options=load_options)

        if print_summary:
            self.print_short_summary()


# TODO: add consistency loss and consistency lambda that are used as a loss function when applying cut_mix or cut_out.
class VoloGAN(CycleGAN):
    def __init__(
        self,
        generator_source_to_target: tf.keras.Model,
        generator_target_to_source: tf.keras.Model,
        discriminator_target: tf.keras.Model,
        discriminator_source: tf.keras.Model,
        lambda_cycle: float = 10.0,
        lambda_identity: float = 0.5,
        lambda_ssim: float = 1.0,
        use_cutmix: bool = False,
        cutmix_probability: float = 0.2,
    ) -> None:
        super(VoloGAN, self).__init__(
            generator_source_to_target,
            generator_target_to_source,
            discriminator_source,
            discriminator_target,
            lambda_cycle,
            lambda_identity,
        )
        self.modelType = "VoloGAN"  # TODO: check how this was used und change to model name and pass to base class
        self.lambda_ssim = lambda_ssim
        self.useCutmix = use_cutmix
        self.cutmixProbability = cutmix_probability

    # To reduce python overhead, and maximize the performance of your TPU, try out the experimental experimental_steps_per_execution argument to Model.compile. Here it increases throughput by about 50%:
    # https://www.tensorflow.org/guide/tpu#train_a_model_using_keras_high_level_apis

    def compile(
        self,
        optimizer_generator_source_to_target: tf.keras.optimizers.Optimizer,
        optimizer_generator_target_to_source: tf.keras.optimizers.Optimizer,
        optimizer_discriminator_source: tf.keras.optimizers.Optimizer,
        optimizer_discriminator_target: tf.keras.optimizers.Optimizer,
        loss_fn_generator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_discriminator_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_cycle_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        loss_fn_identity_loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
        ssim_loss_fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> None:
        super(VoloGAN, self).compile(
            optimizer_generator_source_to_target,
            optimizer_generator_target_to_source,
            optimizer_discriminator_source,
            optimizer_discriminator_target,
            loss_fn_generator_loss,
            loss_fn_discriminator_loss,
            loss_fn_cycle_loss,
            loss_fn_identity_loss,
        )
        self.ssim_loss_fn = ssim_loss_fn

    def train_step(self, train_data_batch: tf.Tensor) -> List[tf.Tensor]:
        real_source, real_target = train_data_batch

        with tf.GradientTape(persistent=True) as tape:
            # Generating fakes
            generated_target = self.generator_source_to_target(real_source, training=True)
            generated_source = self.generator_target_to_source(real_target, training=True)

            # Cycle mapping
            cycled_source = self.generator_target_to_source(generated_target, training=True)
            cycled_target = self.generator_source_to_target(generated_source, training=True)

            # Identity mapping
            same_source = self.generator_target_to_source(real_source, training=True)
            same_target = self.generator_source_to_target(real_target, training=True)

            if DISCRIMINATOR_DESIGN == "PatchGAN":
                # Discriminator output
                disc_real_source = self.discriminator_source(real_source, training=True)
                disc_generated_source = self.discriminator_source(generated_source, training=True)

                disc_real_target = self.discriminator_target(real_target, training=True)
                disc_generated_target = self.discriminator_target(generated_target, training=True)

                # Discriminator loss
                discriminator_source_loss = self.loss_fn_discriminator_loss(disc_real_source, disc_generated_source)
                discriminator_target_loss = self.loss_fn_discriminator_loss(disc_real_target, disc_generated_target)

                # Generator adverserial loss
                generator_source_to_target_loss = self.loss_fn_generator_loss(disc_generated_target)
                generator_target_to_source_loss = self.loss_fn_generator_loss(disc_generated_source)

            elif DISCRIMINATOR_DESIGN == "UNet":
                # Discriminator output
                disc_real_source_encoder, disc_real_source_decoder = self.discriminator_source(
                    real_source, training=True
                )
                disc_generated_source_encoder, disc_generated_source_decoder = self.discriminator_source(
                    generated_source, training=True
                )
                disc_real_target_encoder, disc_real_target_decoder = self.discriminator_target(
                    real_target, training=True
                )
                disc_generated_target_encoder, disc_generated_target_decoder = self.discriminator_target(
                    generated_target, training=True
                )

                if self.useCutmix and np.random.uniform(0, 1) < self.cutmixProbability:
                    batch_size, height, width, channel = rgbd_synthetic.shape
                    mixing_mask = GetCutmixMask(batch_size, height, width, channel)
                    _, cutmix_source = CutMix(
                        real_source, generated_source, ignoreBackground=True, mixing_mask=mixing_mask
                    )
                    _, cutmix_target = CutMix(
                        real_target, generated_target, ignoreBackground=True, mixing_mask=mixing_mask
                    )

                    # calculate discriminator output of cutmix images. Only decoder output relevant!
                    _, disc_cutmix_source = self.discriminator_source(cutmix_source, training=True)
                    _, disc_cutmix_target = self.discriminator_target(cutmix_target, training=True)

                    # calculate cutmix of discriminator output
                    _, cutmix_disc_source = CutMix(
                        disc_real_source_decoder,
                        disc_generated_source_decoder,
                        ignoreBackground=True,
                        mixing_mask=mixing_mask,
                    )
                    _, cutmix_disc_target = CutMix(
                        disc_real_target_decoder,
                        disc_generated_target_decoder,
                        ignoreBackground=True,
                        mixing_mask=mixing_mask,
                    )

                else:
                    disc_cutmix_source = None
                    disc_cutmix_target = None
                    cutmix_disc_source = None
                    cutmix_disc_target = None

                # Discriminator loss
                discriminator_source_loss = self.loss_fn_discriminator_loss(
                    disc_real_source_encoder,
                    disc_real_source_decoder,
                    disc_generated_source_encoder,
                    disc_generated_source_decoder,
                    disc_cutmix_source,
                    cutmix_disc_source,
                )
                discriminator_target_loss = self.loss_fn_discriminator_loss(
                    disc_real_target_encoder,
                    disc_real_target_decoder,
                    disc_generated_target_encoder,
                    disc_generated_target_decoder,
                    disc_cutmix_target,
                    cutmix_disc_target,
                )

                # Generator adverserial loss
                generator_source_to_target_loss = self.loss_fn_generator_loss(
                    disc_generated_target_encoder, disc_generated_target_decoder
                )
                generator_target_to_source_loss = self.loss_fn_generator_loss(
                    disc_generated_source_encoder, disc_generated_source_decoder
                )

            elif DISCRIMINATOR_DESIGN == "OneShot_GAN":
                # Discriminator output
                (
                    disc_real_source_low_level,
                    disc_real_source_layout,
                    disc_real_source_content,
                ) = self.discriminator_source(real_source, training=True)
                (
                    disc_generated_source_low_level,
                    disc_generated_source_layout,
                    disc_generated_source_content,
                ) = self.discriminator_source(generated_source, training=True)

                (
                    disc_real_target_low_level,
                    disc_real_target_layout,
                    disc_real_target_content,
                ) = self.discriminator_target(real_target, training=True)
                (
                    disc_generated_target_low_level,
                    disc_generated_target_layout,
                    disc_generated_target_content,
                ) = self.discriminator_target(generated_target, training=True)

                # Discriminator loss
                discriminator_source_loss = self.loss_fn_discriminator_loss(
                    disc_real_source_low_level,
                    disc_real_source_layout,
                    disc_real_source_content,
                    disc_generated_source_low_level,
                    disc_generated_source_layout,
                    disc_generated_source_content,
                )
                discriminator_target_loss = self.loss_fn_discriminator_loss(
                    disc_real_target_low_level,
                    disc_real_target_layout,
                    disc_real_target_content,
                    disc_generated_target_low_level,
                    disc_generated_target_layout,
                    disc_generated_target_content,
                )

                # Generator adverserial loss
                generator_source_to_target_loss = self.loss_fn_generator_loss(
                    disc_generated_target_low_level, disc_generated_target_layout, disc_generated_target_content
                )
                generator_target_to_source_loss = self.loss_fn_generator_loss(
                    disc_generated_source_low_level, disc_generated_source_layout, disc_generated_source_content
                )

            else:
                raise Exception("Discriminator is not Defined")

            # Generator cycle loss
            cycle_loss_target = self.loss_fn_cycle_loss(real_target, cycled_target) * self.lambda_cycle
            cycle_loss_source = self.loss_fn_cycle_loss(real_source, cycled_source) * self.lambda_cycle

            # Generator identity loss
            identity_loss_target = self.loss_fn_identity_loss(real_target, same_target) * self.lambda_identity
            identity_loss_source = self.loss_fn_identity_loss(real_source, same_source) * self.lambda_identity

            # Generator SSIM Loss
            ssim_loss_target = self.ssim_loss_fn(real_target, cycled_target) * self.lambda_ssim
            ssim_loss_source = self.ssim_loss_fn(real_source, cycled_source) * self.lambda_ssim

            # Total generator loss
            total_gen_loss_target = (
                generator_source_to_target_loss + cycle_loss_target + identity_loss_target + ssim_loss_target
            )
            total_gen_loss_source = (
                generator_target_to_source_loss + cycle_loss_source + identity_loss_source + ssim_loss_source
            )

        # Get the gradients for the generators
        grads_target = tape.gradient(total_gen_loss_target, self.generator_source_to_target.trainable_variables)
        grads_source = tape.gradient(total_gen_loss_source, self.generator_target_to_source.trainable_variables)

        # Get the gradients for the discriminators
        discriminator_source_grads = tape.gradient(
            discriminator_source_loss, self.discriminator_source.trainable_variables
        )
        discriminator_target_grads = tape.gradient(
            discriminator_target_loss, self.discriminator_target.trainable_variables
        )

        # Update the weights of the generators
        self.optimizer_generator_source_to_target.apply_gradients(
            zip(grads_target, self.generator_source_to_target.trainable_variables)
        )
        self.optimizer_generator_target_to_source.apply_gradients(
            zip(grads_source, self.generator_target_to_source.trainable_variables)
        )

        # Update the weights of the discriminators
        self.optimizer_discriminator_source.apply_gradients(
            zip(discriminator_source_grads, self.discriminator_source.trainable_variables)
        )
        self.optimizer_discriminator_target.apply_gradients(
            zip(discriminator_target_grads, self.discriminator_target.trainable_variables)
        )
        return [
            total_gen_loss_target,
            total_gen_loss_source,
            discriminator_target_loss,
            discriminator_source_loss,
            generator_source_to_target_loss,
            generator_target_to_source_loss,
            cycle_loss_target,
            cycle_loss_source,
            identity_loss_target,
            identity_loss_source,
            ssim_loss_target,
            ssim_loss_source,
        ]

    def test_step(self, test_data_batch: tf.Tensor) -> List[tf.Tensor]:
        real_source, real_target = test_data_batch

        # Generate fakes
        generated_target = self.generator_source_to_target(real_source, training=False)
        generated_source = self.generator_target_to_source(real_target, training=False)

        # Cycle mapping
        cycled_source = self.generator_target_to_source(generated_target, training=False)
        cycled_target = self.generator_source_to_target(generated_source, training=False)

        # Identity mapping
        same_source = self.generator_target_to_source(real_source, training=False)
        same_target = self.generator_source_to_target(real_target, training=False)

        if DISCRIMINATOR_DESIGN == "PatchGAN":
            # Discriminator output
            disc_real_source = self.discriminator_source(real_source, training=False)
            disc_generated_source = self.discriminator_source(generated_source, training=False)

            disc_real_target = self.discriminator_target(real_target, training=False)
            disc_generated_target = self.discriminator_target(generated_target, training=False)

            # Discriminator loss
            discriminator_source_loss = self.loss_fn_discriminator_loss(disc_real_source, disc_generated_source)
            discriminator_target_loss = self.loss_fn_discriminator_loss(disc_real_target, disc_generated_target)

            # Generator adverserial loss
            generator_source_to_target_loss = self.loss_fn_generator_loss(disc_generated_target)
            generator_target_to_source_loss = self.loss_fn_generator_loss(disc_generated_source)

        elif DISCRIMINATOR_DESIGN == "UNet":
            # Discriminator output
            disc_real_source_encoder, disc_real_source_decoder = self.discriminator_source(real_source, training=False)
            disc_generated_source_encoder, disc_generated_source_decoder = self.discriminator_source(
                generated_source, training=False
            )
            disc_real_target_encoder, disc_real_target_decoder = self.discriminator_target(real_target, training=False)
            disc_generated_target_encoder, disc_generated_target_decoder = self.discriminator_target(
                generated_target, training=False
            )

            # Discriminator loss -> No Cutmix Regularization during Test!
            discriminator_source_loss = self.loss_fn_discriminator_loss(
                disc_real_source_encoder,
                disc_real_source_decoder,
                disc_generated_source_encoder,
                disc_generated_source_decoder,
                None,
                None,
            )
            discriminator_target_loss = self.loss_fn_discriminator_loss(
                disc_real_target_encoder,
                disc_real_target_decoder,
                disc_generated_target_encoder,
                disc_generated_target_decoder,
                None,
                None,
            )

            # Generator adverserial loss
            generator_source_to_target_loss = self.loss_fn_generator_loss(
                disc_generated_target_encoder, disc_generated_target_decoder
            )
            generator_target_to_source_loss = self.loss_fn_generator_loss(
                disc_generated_source_encoder, disc_generated_source_decoder
            )

        elif DISCRIMINATOR_DESIGN == "OneShot_GAN":
            # Discriminator output

            disc_real_source_low_level, disc_real_source_layout, disc_real_source_content = self.discriminator_source(
                real_source, training=False
            )
            (
                disc_generated_source_low_level,
                disc_generated_source_layout,
                disc_generated_source_content,
            ) = self.discriminator_source(generated_source, training=False)

            disc_real_target_low_level, disc_real_target_layout, disc_real_target_content = self.discriminator_target(
                real_target, training=False
            )
            (
                disc_generated_target_low_level,
                disc_generated_target_layout,
                disc_generated_target_content,
            ) = self.discriminator_target(generated_target, training=False)

            # Discriminator loss
            discriminator_source_loss = self.loss_fn_discriminator_loss(
                disc_real_source_low_level,
                disc_real_source_layout,
                disc_real_source_content,
                disc_generated_source_low_level,
                disc_generated_source_layout,
                disc_generated_source_content,
            )
            discriminator_target_loss = self.loss_fn_discriminator_loss(
                disc_real_target_low_level,
                disc_real_target_layout,
                disc_real_target_content,
                disc_generated_target_low_level,
                disc_generated_target_layout,
                disc_generated_target_content,
            )

            # Generator adverserial loss
            generator_source_to_target_loss = self.loss_fn_generator_loss(
                disc_generated_target_low_level, disc_generated_target_layout, disc_generated_target_content
            )
            generator_target_to_source_loss = self.loss_fn_generator_loss(
                disc_generated_source_low_level, disc_generated_source_layout, disc_generated_source_content
            )

        else:
            raise Exception("Discriminator is not Defined")

        # Generator cycle loss
        cycle_loss_target = self.loss_fn_cycle_loss(real_target, cycled_target) * self.lambda_cycle
        cycle_loss_source = self.loss_fn_cycle_loss(real_source, cycled_source) * self.lambda_cycle

        # Generator identity loss
        identity_loss_target = self.loss_fn_identity_loss(real_target, same_target) * self.lambda_identity
        identity_loss_source = self.loss_fn_identity_loss(real_source, same_source) * self.lambda_identity

        # Generator SSIM Loss
        ssim_loss_target = self.ssim_loss_fn(real_target, cycled_target) * self.lambda_ssim
        ssim_loss_source = self.ssim_loss_fn(real_source, cycled_source) * self.lambda_ssim

        # Total generator loss
        total_gen_loss_target = (
            generator_source_to_target_loss + cycle_loss_target + identity_loss_target + ssim_loss_target
        )
        total_gen_loss_source = (
            generator_target_to_source_loss + cycle_loss_source + identity_loss_source + ssim_loss_source
        )

        return [
            total_gen_loss_target,
            total_gen_loss_source,
            discriminator_target_loss,
            discriminator_source_loss,
            generator_source_to_target_loss,
            generator_target_to_source_loss,
            cycle_loss_target,
            cycle_loss_source,
            identity_loss_target,
            identity_loss_source,
            ssim_loss_target,
            ssim_loss_source,
        ]
