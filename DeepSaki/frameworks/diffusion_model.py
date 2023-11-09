import logging
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from DeepSaki.diffusion.diffusion_process import GaussianDiffusionProcess
from DeepSaki.diffusion.sampler import Sampler
from DeepSaki.losses.diffusion_model_losses import L_VLB
from DeepSaki.losses.diffusion_model_losses import Get_L_VLB_Term
from DeepSaki.losses.diffusion_model_losses import Get_VLB_prior
from DeepSaki.losses.diffusion_model_losses import L_simple
from DeepSaki.tensor_ops.tensor_ops import sample_array_to_tensor
from DeepSaki.types.diffusion_enums import ScheduleType
from DeepSaki.types.diffusion_enums import variance_type

class DiffusionModel:
    """Abstraction of a diffusion model including relevant settings."""

    def __init__(
        self,
        model: tf.keras.Model,
        diffusion_steps: int,
        sampler: Sampler,
        diffusionInputShapeChannels: int,
        diffusionInputShapeHeightWidth: Tuple[int, int],
        beta_schedule_type: ScheduleType = ScheduleType.LINEAR,
        variance_type: variance_type = variance_type.UPPER_BOUND,
        loss_weighting_type: str = "simple",
        lambda_vlb: Optional[float] = None,
        use_mixed_precission: bool = True,
    ) -> None:
        self.model = model
        self.loss_weighting_type = loss_weighting_type
        self.lambda_vlb = lambda_vlb
        self.diffusion_steps = diffusion_steps
        self.sampler = sampler
        self.use_mixed_precission = use_mixed_precission
        self.beta_schedule_type = beta_schedule_type
        self.variance_type = variance_type
        self.diffusionInputShapeChannels = diffusionInputShapeChannels
        self.diffusionInputShapeHeightWidth = diffusionInputShapeHeightWidth
        self.optimizer: tf.keras.optimizers.Optimizer = None
        self.strategy = tf.distribute.get_strategy()
        self.diffusion = GaussianDiffusionProcess(
            betaSchedule=self.beta_schedule_type, variance_type=self.variance_type
        )

    def sample(self):
        return self.sampler.sample()

    def compile(self, optimizer: tf.keras.optimizers.Optimizer) -> None:
        """Setting the optimizer of the diffusion model.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): Optimizer to train the diffusion model
        """
        if self.use_mixed_precission:
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        self.optimizer = optimizer

    # TODO: should this be here? I guess it is because of the Callback I used in rgb-d-fusion
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

    # TODO: check if this might be better in a diffusion loss class
    def _get_loss_weighting(
        self,
        weighting_type: str,
        batch_size: int,
        timestep_values: tf.Tensor,
    ) -> Union[float, tf.Tensor]:
        """Get the value for the loss weighting depending on the weighting_type.

        Args:
            weighting_type (str): [ simple | P2 ]
            batch_size (int): Batchsize of the data
            timestep_values (tf.Tensor): Indicies of the timesteps used to obtain the actual value from the beta schedule.

        Raises:
            ValueError: Unsupported value for self.loss_weighting_type

        Returns:
            Union[float, tf.Tensor]: Loss weighting factor.
        """
        # TODO: loss weighting type should be an enum
        if weighting_type == "simple":
            return 1
        if weighting_type == "P2":
            return sample_array_to_tensor(
                self.diffusion.beta_schedule.lambda_t_tick_simple, timestep_values, shape=(batch_size, 1, 1, 1)
            )
        raise ValueError(f"Undefined loss_weighting_type provided: {self.loss_weighting_type}")

    @tf.function
    def train_step(self, batched_x0: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Performs a single optimization step of the optimizer for a single batch.

        Args:
            batched_x0 (tf.Tensor): Unnoisy data samples at timestep 0.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            tf.Tensor: Loss of the current batch
        """
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        batch_size = batched_x0_diffusion_input.shape[0]

        timestep_values = self.diffusion.draw_random_timestep(batch_size)  # 1 value for each sample in the batch

        batched_xt_diffusion_input, noise = self.diffusion.q_sample_xt(batched_x0_diffusion_input, timestep_values)

        loss_scaling_factor = self._get_loss_weighting(self.loss_weighting_type, batch_size, timestep_values)

        with tf.GradientTape() as tape:
            prediction = self.model(batched_x0_condition, batched_xt_diffusion_input, timestep_values)

            if self.variance_type in [variance_type.LEARNED, variance_type.LEARNED_RANGE]:
                # split prediction into noise and var
                # recombine it with applying tf.stop_gradient to the noise value
                # feed it to the Get_L_VLB_Term method
                pred_noise, pred_var = tf.split(prediction, 2, axis=-1)
                pred_stopped_noise = tf.concat((tf.stop_gradient(pred_noise), pred_var), axis=-1)
                loss_simple = L_simple(noise, pred_noise, global_batchsize, loss_scaling_factor)
                loss_vlb = self.lambda_vlb * L_VLB(
                    pred_stopped_noise,
                    batched_x0_diffusion_input,
                    batched_xt_diffusion_input,
                    timestep_values,
                    self.diffusion,
                    global_batchsize,
                )
                loss = loss_simple + loss_vlb
            else:
                loss = L_simple(noise, prediction, global_batchsize, loss_scaling_factor)

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
    def test_step(self, batched_x0: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Performs a single test step without updating the weights for a single batch.

        Args:
            batched_x0 (tf.Tensor): Unnoisy data samples at timestep 0.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            tf.Tensor: Loss of the current batch
        """
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        batch_size = batched_x0_diffusion_input.shape[0]

        timestep_values = self.diffusion.draw_random_timestep(batch_size)  # 1 value for each sample in the batch

        batched_xt_diffusion_input, noise = self.diffusion.q_sample_xt(batched_x0_diffusion_input, timestep_values)

        loss_scaling_factor = self._get_loss_weighting(self.loss_weighting_type, batch_size, timestep_values)

        prediction = self.model(batched_x0_condition, batched_xt_diffusion_input, timestep_values, training=False)

        if self.variance_type in [variance_type.LEARNED, variance_type.LEARNED_RANGE]:
            # split prediction into noise and var
            # recombine it with applying tf.stop_gradient to the noise value
            # feed it to the Get_L_VLB_Term method
            pred_noise, pred_var = tf.split(prediction, 2, axis=-1)
            pred_stopped_noise = tf.concat((tf.stop_gradient(pred_noise), pred_var), axis=-1)
            loss_simple = L_simple(noise, pred_noise, global_batchsize, loss_scaling_factor)
            loss_vlb = self.lambda_vlb * L_VLB(
                pred_stopped_noise,
                batched_x0_diffusion_input,
                batched_xt_diffusion_input,
                timestep_values,
                self.diffusion,
                global_batchsize,
            )
            loss = loss_simple + loss_vlb
        else:
            loss = L_simple(noise, prediction, global_batchsize, loss_scaling_factor)

        return loss

    # TODO: eval is very specific for the data structure, e.g. images... probably s.th. more generic would be better? e.g. only VLB and then a list of functions to eval
    # @tf.function
    def eval_step(self, batched_x0: tf.Tensor, global_batchsize: int, threshold: float = -0.9) -> Tuple[float, ...]:
        """Performs a single evaluation step without updating the weights for a single batch.

        Args:
            batched_x0 (tf.Tensor): Unnoisy data samples at timestep 0.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.
            threshold (float, optional): Threshold values to remove back ground when calculating pixel-wise distance
                metrics e.g. MAE or MSE. Defaults to -0.9.

        Returns:
            Tuple[float, ...]: Calculated metrics are: VLB, mae, mse, iou, dice, x_translation, y_translation,
                mae_shifted, mse_shifted, iou_shifted, dice_shifted
        """
        batched_x0_condition, batched_x0_diffusion_input = batched_x0
        inputShape = batched_x0_diffusion_input.shape
        VLB_terms = []

        xt_diffusion_input_reverse = tf.random.normal(inputShape)

        for i in tqdm(range(self.diffusion_steps), ncols=100):
            t = np.expand_dims(np.array(self.diffusion_steps - i - 1, np.int32), 0)
            model_prediction = self.model(batched_x0_condition, xt_diffusion_input_reverse, t, training=False)
            xt_diffusion_input_forward, _ = self.diffusion.q_sample_xt(batched_x0_diffusion_input, t)
            # update xt_diffusion_input_reverse for next cycle by sampling diffusion_input from distr.
            xt_diffusion_input_reverse = self.diffusion.p_sample_xtm1_given_xt(
                xt_diffusion_input_reverse, model_prediction, t
            )

            # mean reduction of batches is performed later together with the summation of L0,L1,L2,L3 ...
            vlb_term = Get_L_VLB_Term(
                model_prediction,
                batched_x0_diffusion_input,
                xt_diffusion_input_forward,
                t,
                self.diffusion,
                clip_denoised=True,
                return_bits=True,
            )
            VLB_terms.append(vlb_term)

        kl_prior = Get_VLB_prior(batched_x0_diffusion_input, self.diffusion)
        VLB_terms.append(kl_prior)

        # VLB is sum of individual losses of each timestep
        # Sum all individual loss terms L0, L1, ..., LT. CAUTION: Shape of VLB_terms is (timestep, batch)! -> sum axis 0 for loss summation
        VLB = tf.math.reduce_sum(VLB_terms, axis=0)

        # reduce mean operation considering potential distributed training strategy
        if threshold is not None:
            xt_diffusion_input_reverse = tf.where(
                xt_diffusion_input_reverse < threshold, -1.0, xt_diffusion_input_reverse
            )  # Must be 1.0 not -1 so the output is not casted as int...

        # shape of iou and dice is [batch] -> one score for each batch
        # TODO: check cdd_helper
        metrics = cdd_helper.calc_depth_metrics(
            depth_gt=batched_x0_diffusion_input, depth_pred=xt_diffusion_input_reverse, threshold=threshold
        )
        y_shifts, x_shifts, depth_pred_shifted = cdd_helper.get_shift(
            batched_x0_diffusion_input, xt_diffusion_input_reverse, threshold=threshold
        )
        metrics_shifted = cdd_helper.calc_depth_metrics(
            depth_gt=batched_x0_diffusion_input, depth_pred=depth_pred_shifted, threshold=threshold
        )

        # perform manual loss reduction over batch axis
        VLB = (1.0 / global_batchsize) * tf.math.reduce_sum(VLB)
        mae = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics["mae"])
        mse = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics["mse"])
        iou = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics["iou"])
        dice = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics["dice"])
        x_translation = (1.0 / global_batchsize) * tf.math.reduce_sum(tf.math.abs(y_shifts))
        y_translation = (1.0 / global_batchsize) * tf.math.reduce_sum(tf.math.abs(x_shifts))
        mae_shifted = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics_shifted["mae"])
        mse_shifted = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics_shifted["mse"])
        iou_shifted = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics_shifted["iou"])
        dice_shifted = (1.0 / global_batchsize) * tf.math.reduce_sum(metrics_shifted["dice"])

        logging.info(
            f"VLB: {VLB:.4f} MAE: {mae:.4f} MSE: {mse:.4f} IoU: {iou:.4f} Dice: {dice:.4f} x_translation: {x_translation:.4f} y_translation: {y_translation:.4f} MAE shifted: {mae_shifted:.4f} MSE shifted: {mse_shifted:.4f} IoU shifted: {iou_shifted:.4f}  Dice shifted: {dice_shifted:.4f}"
        )
        logging.info("-------------------------------------------------")

        return (
            VLB,
            mae,
            mse,
            iou,
            dice,
            x_translation,
            y_translation,
            mae_shifted,
            mse_shifted,
            iou_shifted,
            dice_shifted,
        )

    @tf.function
    def distributed_train_step(self, batch_train: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Distributes the training step on all available workers.

        Args:
            batch_train (tf.Tensor): Current batch of training data.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            tf.Tensor: Tensor containing the reduced (summation) losses from all workers.
        """
        per_replica_loss = self.strategy.run(
            self.train_step,
            args=(
                batch_train,
                global_batchsize,
            ),
        )
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, axis=None)

    @tf.function
    def distributed_test_step(self, batch_test: tf.Tensor, global_batchsize: int) -> tf.Tensor:
        """Distributes the testing step on all available workers.

        Args:
            batch_test (tf.Tensor): Current batch of training data.
            global_batchsize (int): Batch size considering all workers running in parallel in a data parallel setup.

        Returns:
            tf.Tensor: Tensor containing the reduced (summation) losses from all workers.
        """
        per_replica_loss = self.strategy.run(
            self.test_step,
            args=(
                batch_test,
                global_batchsize,
            ),
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
        per_replica_metric_vector = self.strategy.run(
            self.eval_step,
            args=(
                batch_test,
                global_batchsize,
            ),
        )

        # The loop iterates over loss values, not over replicas!!!
        # i.e. eval_step() returns [metric1, metric2, metric3] from each replica, so each loss element in the
        # final output is obtained by summing metric1 from all replicas, summing metric2 from all replicas, etc.
        reduced_metric_vector = []
        for per_replica_metric in per_replica_metric_vector:
            reduced_metric_vector.append(
                self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_metric, axis=None)
            )

        return reduced_metric_vector
