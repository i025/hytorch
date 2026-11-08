is_simple_core = False

if is_simple_core:
    from hytorch.core_simple import Variable
    from hytorch.core_simple import Function
    from hytorch.core_simple import using_config
    from hytorch.core_simple import no_grad
    from hytorch.core_simple import as_array
    from hytorch.core_simple import as_variable
    from hytorch.core_simple import setup_variable
else:
    from hytorch.core import Variable
    from hytorch.core import Function
    from hytorch.core import using_config
    from hytorch.core import no_grad
    from hytorch.core import as_array
    from hytorch.core import as_variable
    from hytorch.core import setup_variable
    from hytorch.core import Parameter
    from hytorch.core import Config  # #
    from hytorch.layers import Layer      # from hytorch import Layer
    from hytorch.models import Model
    from hytorch import functions  # #
    from hytorch.dataloaders import DataLoader
    from hytorch import cuda  # #
    from hytorch.core import Config
    from hytorch.core import test_mode

    import hytorch.datasets
    import hytorch.dataloaders
    import hytorch.optimizers
    import hytorch.functions
    import hytorch.functions_conv
    import hytorch.layers
    import hytorch.utils
    import hytorch.cuda
    import hytorch.transforms
    import hytorch.augmentations
    import hytorch.utils_data
    from hytorch.utils_data import DatasetAnalyzer, analyze_dataset
    from hytorch import onnx
    from hytorch import quantization
setup_variable()
