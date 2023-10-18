"""Loss functions designed for image like data of the shape (`batch`, `height`, `width`, `channels`)"""
import numpy as np
import tensorflow as tf


class PixelDistanceLoss(tf.keras.losses.Loss):
    """calculates a pixel distance loss (per pixel loss) of two images of the shape (batch, height, width, channels)."""

    def __init__(
        self,
        global_batch_size: int,
        calculation_type: str = "per_image",
        normalize_depth_channel: bool = False,
        loss_type: str = "mae",
        loss_reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
    ) -> None:
        """Dunder method to initialize `PixelDistanceLoss`.

        Args:
            global_batch_size (int): Batch size considering all workers running in parallel in a data parallel setup
            calculation_type (str, optional): Determines how the loss is calculated: ["per_image" | "per_channel"].
                Defaults to "per_image".
            normalize_depth_channel (bool, optional): For RGBD images, the weight of depth is increased by multiplying
                the depth by the number of color channels. Defaults to False.
            loss_type (str, optional): Loss to apply: ["mae" | "mse"]. Defaults to "mae".
            loss_reduction (tf.keras.losses.Reduction, optional): Determines how the loss is reduced. Defaults to
                tf.keras.losses.Reduction.AUTO.
        """
        super(PixelDistanceLoss, self).__init__(reduction=loss_reduction)
        self.global_batch_size = global_batch_size
        self.calculation_type = calculation_type
        self.normalize_depth_channel = normalize_depth_channel
        self.loss_type = loss_type

    def call(self, img1: tf.Tensor, img2: tf.Tensor) -> tf.Tensor:
        """Calculates the pixel distance between `img1` and `img2`.

        Args:
            img1 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).
            img2 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).

        Raises:
            ValueError: if `self.loss_type` is not a valid option.
            ValueError: if `self.calculation_type` is not a valid option.

        Returns:
            Tensor containing the loss defined in `loss_type`.
        """
        img1 = tf.cast(img1, tf.dtypes.float32)
        img2 = tf.cast(img2, tf.dtypes.float32)
        loss = 0.0

        if self.loss_type == "mae":
            error_func = tf.abs
        elif self.loss_type == "mse":
            error_func = tf.square
        else:
            raise ValueError(f"Parameter loss_type={self.loss_type} is not defined. Use 'mae' or 'mse' instead.")

        if self.calculation_type == "per_channel":
            # initialize all weights with 1
            channel_weight = np.ones(img1.shape[-1])
            if self.normalize_depth_channel:
                # set weight of the depth channel according to the number of color channels: e.g. for RGB = 3
                channel_weight[-1] = len(channel_weight) - 1
                for i in range(img1.shape[-1]):
                    loss += (
                        channel_weight[i]
                        * tf.reduce_mean(error_func(img1[:, :, :, i] - img2[:, :, :, i]))
                        * (1.0 / self.global_batch_size)
                    )

        elif self.calculation_type == "per_image":
            loss = tf.reduce_mean(error_func(img1 - img2)) * (1.0 / self.global_batch_size)

        else:
            raise ValueError(f"Pixel distance type '{self.calculation_type}' is not defined.")

        return loss


class StructuralSimilarityLoss(tf.keras.losses.Loss):
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
        super(StructuralSimilarityLoss, self).__init__(reduction=loss_reduction)
        self.global_batch_size = global_batch_size
        self.calculation_type = calculation_type
        self.normalize_depth_channel = normalize_depth_channel
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
        return (1 - ssim) * (1.0 / self.global_batch_size)

    def call(self, img1: tf.Tensor, img2: tf.Tensor) -> tf.Tensor:
        """Calculates the SSIM loss between `img1` and `img2`.

        Args:
            img1 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).
            img2 (tf.Tensor): Image of shape (`batch_size`, `height`, `width`,`channel`).

        Raises:
            ValueError: if `self.calculation_type` is not a valid option.

        Returns:
            Tensor containing the loss value.
        """
        img1 = tf.cast(img1, tf.dtypes.float32)
        img2 = tf.cast(img2, tf.dtypes.float32)
        ssim_loss = 0.0

        if self.calculation_type == "per_image":
            ssim_loss = self._calc_ssim_loss(img1, img2)
        elif self.calculation_type == "per_channel":
            # initialize all weights with 1
            channel_weight = np.ones(img1.shape[-1])
            if self.normalize_depth_channel:
                # set weight of the depth channel according to the number of color channels: e.g. for RGB = 3
                channel_weight[-1] = len(channel_weight) - 1
            # loop over all channels of the image
            for i in range(img1.shape[-1]):
                ssim_loss += channel_weight[i] * self._calc_ssim_loss(img1[:, :, :, i], img2[:, :, :, i])
        else:
            raise ValueError(f"ssim calculation type '{self.calculation_type}' is not defined")

        return ssim_loss
