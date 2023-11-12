from typing import List

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DeepSaki.diffusion.diffusion_process import GaussianDiffusionProcess
from DeepSaki.losses.diffusion_model_losses import DiffusionLoss
from DeepSaki.types.diffusion_enums import variance_type


# TODO: For now it is an image conditioned Diffusion model. In a future that should change to be able to support unconditional or token conditional diffuision
class DiffusionModel:
    """Abstraction of a diffusion model including relevant settings."""

    def __init__(
        self,
        model: tf.keras.Model,
        diffusion_process: GaussianDiffusionProcess,
        diffusion_loss: DiffusionLoss,
        diffusion_input_shape: List[int],
    ) -> None:
        """Initializes the `DiffusionModel` object.

        Args:
            model (tf.keras.Model): Implementation of the model. The model is expected to have 3 Inputs:
                Condition, Diffusion and timestep.
            diffusion_process (GaussianDiffusionProcess): Diffusion Process object.
            diffusion_loss (DiffusionLoss): DiffusionLoss object.
            diffusion_input_shape (List[int]): Shape of the diffusion input. Used to generate the initial condition.
        """
        self.model = model
        self.diffusion_loss = diffusion_loss
        self.diffusion_input_shape = diffusion_input_shape
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.strategy = tf.distribute.get_strategy()
        self.diffusion_process = diffusion_process

        policy = tf.keras.mixed_precision.global_policy()
        self.use_mixed_precission = policy.name in ["mixed_float16", "mixed_bfloat16"]

    def compile(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """Setting the optimizer of the diffusion model.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to train the diffusion model
        """
        if self.use_mixed_precission:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        self.optimizer = optimizer

    # TODO: should this be here? I guess it is because of the Callback I used in rgb-d-fusion. Could be moved into the optimizer api as a helper.
    def get_current_learning_rate(self) -> float:
        """Get the currently set learning rate from the optimizer.

        Returns:
            float: Current learning rate.
        """
        # _current_learning_rate considers also updates performed by learning rate schedules
        # LossScaleOptimizer is a Wrapper Object -> inner_optimizer gets the actual one.
        optimizer = self.optimizer.inner_optimizer if self.use_mixed_precission else self.optimizer

        if isinstance(optimizer.lr, tf.keras.optimizers.schedules.LearningRateSchedule):
            current_lr = optimizer.lr(optimizer.iterations)
        else:
            current_lr = optimizer.lr
        return current_lr

    @tf.function
    def train_step(self, x0: tf.Tensor) -> tf.Tensor:
        """Performs a single optimization step of the optimizer for a single batch.

        Args:
            x0 (tf.Tensor): Unnoisy data samples at timestep 0.

        Returns:
            loss: Training loss of the current batch.
        """
        x0_condition, x0_diffusion = x0

        xt_diffusion, timestep, noise = self.diffusion_process.get_random_sample_forward_diffusion(x0_diffusion)

        with tf.GradientTape() as tape:
            prediction = self.model(x0_condition, xt_diffusion, timestep)

            if self.diffusion_process.variance_type in [variance_type.LEARNED, variance_type.LEARNED_RANGE]:
                loss = self.diffusion_loss.hybrid_loss(
                    noise,
                    prediction,
                    x0_diffusion,
                    xt_diffusion,
                    timestep,
                )
            else:
                loss = self.diffusion_loss.simple_loss(noise, prediction, timestep)

            if self.use_mixed_precission:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if self.use_mixed_precission:
            scaled_grads = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_grads)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    @tf.function
    def test_step(self, x0: tf.Tensor) -> tf.Tensor:
        """Performs a single test step without updating the weights for a single batch.

        Args:
            x0 (tf.Tensor): Unnoisy data samples at timestep 0.

        Returns:
            loss: Test loss of the current batch
        """
        x0_condition, x0_diffusion = x0

        xt_diffusion, timestep, noise = self.diffusion_process.get_random_sample_forward_diffusion(x0_diffusion)

        prediction = self.model(x0_condition, xt_diffusion, timestep, training=False)

        if self.diffusion_process.variance_type in [variance_type.LEARNED, variance_type.LEARNED_RANGE]:
            loss = self.diffusion_loss.hybrid_loss(
                noise,
                prediction,
                x0_diffusion,
                xt_diffusion,
                timestep,
            )
        else:
            loss = self.diffusion_loss.simple_loss(noise, prediction)

        return loss

    # @tf.function
    def eval_step(self, x0: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Performs a single evaluation step without updating the weights for a single batch.

        Args:
            x0 (tf.Tensor): Unnoisy data samples at timestep 0.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            vlb: variational lower bound.
        """
        x0_condition, x0_diffusion = x0
        input_shape = x0_diffusion.shape
        vlb_terms_list = []

        xt_diffusion_reverse = tf.random.normal(input_shape)
        diffusion_steps = self.diffusion_process.diffusion_steps

        for i in tqdm(range(diffusion_steps), ncols=100):
            t = np.expand_dims(np.array(diffusion_steps - i - 1, np.int32), 0)
            model_prediction = self.model(x0_condition, xt_diffusion_reverse, t, training=False)
            xt_diffusion_forward, _ = self.diffusion_process.q_sample_xt(x0_diffusion, t)
            # update xt_diffusion_reverse for next cycle by sampling diffusion_input from distr.
            xt_diffusion_reverse = self.diffusion_process.p_sample_xtm1_given_xt(
                xt_diffusion_reverse, model_prediction, t
            )

            # mean reduction of batches is performed later together with the summation of L0,L1,L2,L3 ...
            vlb_term = self.diffusion_loss.get_vlb_loss_term(model_prediction, x0_diffusion, xt_diffusion_forward, t)
            vlb_terms_list.append(vlb_term)

        kl_divergence_prior = self.diffusion_loss.get_vlb_prior(x0_diffusion)
        vlb_terms_list.append(kl_divergence_prior)

        # VLB is sum of individual losses of each timestep
        # Sum all individual loss terms L0, L1, ..., LT. CAUTION: Shape of VLB_terms is (timestep, batch)! -> sum axis 0 for loss summation
        vlb = tf.math.reduce_sum(vlb_terms_list, axis=0)

        # perform manual loss reduction over batch axis
        vlb = (1.0 / global_batchsize) * tf.math.reduce_sum(vlb)

        return vlb

    @tf.function
    def distributed_train_step(self, batch_train: tf.Tensor) -> tf.Tensor:
        """Distributes the training step on all available workers.

        Args:
            batch_train (tf.Tensor): Current batch of training data.

        Returns:
            tf.Tensor: Tensor containing the reduced (summation) losses from all workers.
        """
        per_replica_loss = self.strategy.run(
            self.train_step,
            args=(batch_train,),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def distributed_test_step(self, batch_test: tf.Tensor) -> tf.Tensor:
        """Distributes the testing step on all available workers.

        Args:
            batch_test (tf.Tensor): Current batch of training data.

        Returns:
            tf.Tensor: Tensor containing the reduced (summation) losses from all workers.
        """
        per_replica_loss = self.strategy.run(
            self.test_step,
            args=(batch_test,),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    # @tf.function
    def distributed_eval_step(self, batch_test: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Distributes the evaluation step on all available workers.

        Args:
            batch_test (tf.Tensor): Current batch of training data.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            tf.Tensor: Tensor containing the reduced (summation) metrics from all workers.
        """
        per_replica_metric = self.strategy.run(
            self.eval_step,
            args=(batch_test, global_batchsize),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)


# TODO: probably some sort of DiffusionMetrics Class?
# TODO: VLB would alwas be calculated and then some others could be optional
# class MetricRegister:
#     def __init__(
#         self,
#         global_batchsize: int,
#         metric_funcs: Optional[Dict[str, Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]],
#     ):
#         self.metric_funcs = metric_funcs
#         self.global_batchsize = global_batchsize

#     def calc_all_metrics(self, model_input: tf.Tensor, model_prediction: tf.Tensor) -> Dict[str, tf.Tensor]:
#         output = {}
#         # 1. Calc VLB

#         # 2. Calc all other metrics
#         for key, func in self.metric_funcs.items():
#             output[key] = (1.0 / self.global_batchsize) * func(model_input, model_prediction)

#         return output
