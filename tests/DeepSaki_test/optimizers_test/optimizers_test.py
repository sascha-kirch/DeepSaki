import inspect
import os

import pytest

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # deactivate tensorflow warnings and infos. Keep Errors

from tensorflow.python.keras.optimizer_v2 import optimizer_v2

from DeepSaki.optimizers.swats import SwatsAdam
from DeepSaki.optimizers.swats import SwatsNadam

@pytest.mark.parametrize(
    "optimizer_object",
    [
        SwatsAdam(),
        SwatsNadam(),
    ],
)
class TestGenericOptimizer:
    def test_get_dict(self, optimizer_object):
        config = optimizer_object.get_config()
        # func bellow gets all variable names of the __init__ param list. [1::] removes "self" from that list.
        expected_keys = inspect.getfullargspec(optimizer_object.__init__)[0][1::]
        key_in_config = [key in config for key in expected_keys]
        assert all(key_in_config), f"not all expected keys found in config: {key_in_config}"

    def test_layer_is_subclass_of_tensorflow_layer(self, optimizer_object):
        assert isinstance(optimizer_object, optimizer_v2.OptimizerV2)
