"""Loss functions designed for image like data of the shape (`batch`, `height`, `width`, `channels`)."""
from abc import ABC
from abc import abstractmethod
from typing import Callable

import numpy as np
import tensorflow as tf


class ImageBasedLoss(tf.keras.losses.Loss, ABC):
    """Abstract base class for image based losses.

    Sub-classes must override `_error_func(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor`, the actuall
    loss function that callculates the loss between two tensors.
    """

    def __init__(
        self,
        global_batch_size: int,
        calculation_type: str = "per_image",
        normalize_depth_channel: bool = False,
        loss_reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
    ) -> None:
        """Initializes the abstract base class `ImageBasedLoss`.

        Args:
            global_batch_size (int): Batch size considering all workers running in parallel in a data parallel setup
            calculation_type (str, optional): Determines how the loss is calculated: ["per_image" | "per_channel"].
                Defaults to "per_image".
            normalize_depth_channel (bool, optional): For RGBD images, the weight of depth is increased by multiplying
                the depth by the number of color channels. Defaults to False.
            loss_reduction (tf.keras.losses.Reduction, optional): Determines how the loss is reduced. Defaults to
                tf.keras.losses.Reduction.AUTO.

        Raises:
            ValueError: if `calculation_type` is not a valid option.
        """
        super(ImageBasedLoss, self).__init__(reduction=loss_reduction)

        match calculation_type:
            case "per_channel":
                self.loss_calc_func = self._calc_loss_per_channel
            case "per_image":
                self.loss_calc_func = self._calc_loss_per_image
            case _:
                raise ValueError(f"Pixel distance type '{calculation_type}' is not defined.")

        self.global_batch_size = global_batch_size
        self.normalize_depth_channel = normalize_depth_channel

    @abstractmethod
    def _error_func(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        pass

    def _get_channel_weights(self, img_shape: tf.TensorShape, normalize_last_channel: bool) -> np.ndarray:
        channel_weight = np.ones(img_shape[-1])
        if normalize_last_channel:
            # set weight of the depth channel according to the number of color channels: e.g. for RGB = 3
            channel_weight[-1] = len(channel_weight) - 1
        return channel_weight

    def _calc_loss_per_channel(
        self,
        img1: tf.Tensor,
        img2: tf.Tensor,
        error_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        # initialize all weights with 1
        channel_weight = self._get_channel_weights(img1.shape, self.normalize_depth_channel)
        loss = 0.0
        for i in range(img1.shape[-1]):
            loss += channel_weight[i] * error_func(img1[:, :, :, i], img2[:, :, :, i]) * (1.0 / self.global_batch_size)
        return loss

    def _calc_loss_per_image(
        self,
        img1: tf.Tensor,
        img2: tf.Tensor,
        error_func: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
    ) -> tf.Tensor:
        return error_func(img1, img2) * (1.0 / self.global_batch_size)

    def call(self, img1: tf.Tensor, img2: tf.Tensor) -> tf.Tensor:
        """Calculates the image loss between `img1` and `img2` using the overriden `error_func` from the sub-class.

        Args:
            img1 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).
            img2 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).

        Returns:
            Tensor containing the loss.
        """
        img1 = tf.cast(img1, tf.dtypes.float32)
        img2 = tf.cast(img2, tf.dtypes.float32)

        return self.loss_calc_func(img1, img2, self._error_func)


class PixelDistanceLoss(ImageBasedLoss):
    """Calculates a pixel distance loss (per pixel loss) of two images of the shape (batch, height, width, channels).

    The distance is either a mean squared error (MSE) or a mean absolute error (MAE).
    """

    def __init__(
        self,
        global_batch_size: int,
        calculation_type: str = "per_image",
        normalize_depth_channel: bool = False,
        loss_type: str = "mae",
        loss_reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
    ) -> None:
        """Initializes the `PixelDistanceLoss`.

        Args:
            global_batch_size (int): Batch size considering all workers running in parallel in a data parallel setup
            calculation_type (str, optional): Determines how the loss is calculated: ["per_image" | "per_channel"].
                Defaults to "per_image".
            normalize_depth_channel (bool, optional): For RGBD images, the weight of depth is increased by multiplying
                the depth by the number of color channels. Defaults to False.
            loss_type (str, optional): Loss to apply: ["mae" | "mse"]. Defaults to "mae".
            loss_reduction (tf.keras.losses.Reduction, optional): Determines how the loss is reduced. Defaults to
                tf.keras.losses.Reduction.AUTO.

        Raises:
            ValueError: if `loss_type` is not a valid option.
        """
        match loss_type:
            case "mae":
                self.error_func_type = tf.abs
            case "mse":
                self.error_func_type = tf.square
            case _:
                raise ValueError(f"Parameter loss_type={loss_type} is not defined. Use 'mae' or 'mse' instead.")

        super(PixelDistanceLoss, self).__init__(
            global_batch_size=global_batch_size,
            calculation_type=calculation_type,
            normalize_depth_channel=normalize_depth_channel,
            loss_reduction=loss_reduction,
        )

    # Override abstract method
    def _error_func(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(self.error_func_type(tensor1 - tensor2))


class StructuralSimilarityLoss(ImageBasedLoss):
    r"""Calculates the structural similarity (SSIM) loss of two images of the shape (batch, height, width, channels).

    The structural similarity loss between two images $x$ and $y$ is defined as: $\mathcal{L}_{SSIM}=1-SSIM(x,y)$.

    SSIM compares contrast, luminance, and structure of two images using statistical parameters i.e., the mean $\mu$,
    the variance $\sigma$, and the covariance $\sigma_{x,y}$ of both images. The SSIM of two images $x$ and $y$ is
    defined as:

    $$
    SSIM(x,y)=
    \underbrace{\left[ \frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2 +C_1}\right]^{\alpha}}_\text{contrast} \cdot
    \underbrace{\left[ \frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2 +C_2}\right]^{\beta}}_\text{luminance} \cdot
    \underbrace{\left[ \frac{\sigma_{x,y}+C_3}{\sigma_x\sigma_y+C_3}\right]^{\gamma}}_\text{structure},
    $$

    where $\alpha$, $\beta$ and $\gamma$ are hyperparameters to give relative importance to individual terms and $C_1$,
    $C_2$ and $C_3$ are constants that must be chosen.
    """

    def __init__(
        self,
        global_batch_size: int,
        calculation_type: str = "per_image",
        normalize_depth_channel: bool = False,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        c1: float = 0.0001,
        c2: float = 0.0009,
        loss_reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
    ) -> None:
        """Dunder method to initialize `StructuralSimilarityLoss`.

        Args:
            global_batch_size (int): Batch size considering all workers running in parallel in a data parallel setup.
            calculation_type (str, optional): Determines how the loss is calculated: ["per_image" | "per_channel"].
                Defaults to "per_image".
            normalize_depth_channel (bool, optional): For RGBD images, the weight of depth is increased by multiplying
                the depth by the number of color channels. Defaults to False.
            alpha (float, optional): Weighting factor for contrast. Defaults to 1.0.
            beta (float, optional): Weighting factor for luminance. Defaults to 1.0.
            gamma (float, optional): Weighting factor for structure. Defaults to 1.0.
            c1 (float, optional): Constant considered in contrast calculation. Defaults to 0.0001.
            c2 (float, optional): Constant considered in luminance calculation. Defaults to 0.0009.
            loss_reduction (tf.keras.losses.Reduction, optional): Determines how the loss is reduced. Defaults to
                tf.keras.losses.Reduction.AUTO.
        """
        super(StructuralSimilarityLoss, self).__init__(
            global_batch_size=global_batch_size,
            calculation_type=calculation_type,
            normalize_depth_channel=normalize_depth_channel,
            loss_reduction=loss_reduction,
        )

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.c1 = c1
        self.c2 = c2
        self.c3 = c2 / 2

    def _calc_ssim_loss(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        """Calculates structural similarity loss between `tensor1` and `tensor2`.

        Args:
            tensor1 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).
            tensor2 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).

        Returns:
            Structural similarity loss between `tensor1` and `tensor2`.
        """
        mu1 = tf.reduce_mean(tensor1)  # mean
        mu2 = tf.reduce_mean(tensor2)
        sigma1 = tf.reduce_mean((tensor1 - mu1) ** 2) ** 0.5  # standard deviation
        sigma2 = tf.reduce_mean((tensor2 - mu2) ** 2) ** 0.5
        covar = tf.reduce_mean((tensor1 - mu1) * (tensor2 - mu2))  # covariance

        luminance = (2 * mu1 * mu2 + self.c1) / (mu1**2 + mu2**2 + self.c1)
        contrast = (2 * sigma1 * sigma2 + self.c2) / (sigma1**2 + sigma2**2 + self.c2)
        structure = (covar + self.c3) / (sigma1 * sigma2 + self.c3)

        ssim = luminance**self.alpha * contrast**self.beta * structure**self.gamma
        return 1 - ssim

    # Override abstract method
    def _error_func(self, tensor1: tf.Tensor, tensor2: tf.Tensor) -> tf.Tensor:
        return self._calc_ssim_loss(tensor1, tensor2)
