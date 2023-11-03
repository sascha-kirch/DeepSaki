
# Changelog

## :rocket: v1.0.0 Nov 5, 2023
GitHub Release Tag: [v1.0.0](https://github.com/sascha-kirch/DeepSaki/releases/tag/v1.0.0)

**Changes:**

- **Framework:**
    - Upgrade from python 3.8 to python 3.10 since it supports a wide range of tensorflow version(2.6-2.14)
    - update of project structure to a modern SW project
    - automated CI/CD
    - docstrings and type annotations
    - initial automated documentation using mkdocs and mike supporting multiple version. Will be further improved in a future.
    - automated code coverage reports
    - automated publishing of packages on release
    - add changelog and contribution documentation
    - providing a dockerfile for the dev environment

- **DeepSaki.initializers:**
    - renamed DeepSaki.initializer -> DeepSaki.initializers
    - Refactored function make_initializer_complex() into a class called ComplexInitializer

- **DeepSaki.layers:**
    - renamed ResidualIdentityBlock -> ResidualBlock
    - New Layers:
        - LearnedPooling
        - FFT3D
        - iFFT3D

- **DeepSaki.losses:**
    - New abstract base class for image based losses.

- **DeepSaki.optimizers:**
    - renamed module from optimizer -> optimizers
    - new method switch_optimizer() to change the current optimizer of SwatsAdam or SwatsNadam optimizer.

- **DeepSaki.augmentations:**
    - renamed module from regularizations to augmentations since it only contained augmentations.

- **DeepSaki.utils:**
    - renamed DetectHw() -> detect_accelerator()
    - renamed EnableXlaAcceleration() -> enable_xla_acceleration()
    - renamed EnableMixedPrecision() -> enable_mixed_precision()

- **New modules:**
    - DeepSaki.types: provides type definitions and enums used in the package.

- **Misc:**
    - general code cleaning and refactoring.
    - changed print expressions to logging
    - using pyproject.toml for project config

- **Outlook:**
    - improved changelogs with issue tickets
    - new modules are planned, e.g. a diffusion module and a framework module featuring CycleGAN and DDPM.


## v0.1.3 May 28, 2022
GitHub Release Tag: [v0.1.3](https://github.com/sascha-kirch/DeepSaki/releases/tag/v0.1.3)

**Changes:**

- FourierConvolution2D() does now consider the circular convolution.

## v0.1.2 May 8, 2022
GitHub Release Tag: [v0.1.2](https://github.com/sascha-kirch/DeepSaki/releases/tag/v0.1.2)

**Changes:**

- Add new layers working with fourier transformations
- Minor code cleaning

## v0.1.0 Dec 23, 2021
GitHub Release Tag: [v0.1.0](https://github.com/sascha-kirch/DeepSaki/releases/tag/v0.1.0)

**Changes:**

- Initial Release
