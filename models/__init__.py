from .ilt_model import PretrainModel, FinetuneModel
from .lithosimul_model import LithoSimulModel


def get_option_setter(opt):
    """Return the static method <modify_commandline_options> of the model class."""
    if opt.phase == 'pretrain':
        return PretrainModel.modify_commandline_options
    elif opt.phase == 'finetune':
        return FinetuneModel.modify_commandline_options
    else:
        return LithoSimulModel.modify_commandline_options

def create_model(opt):
    """Create a model given the option.

    This function warps the class CustomDatasetDataLoader.
    This is the main interface between this package and 'train.py'/'test.py'

    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    if opt.phase == 'pretrain':
        model = PretrainModel
    elif opt.phase == 'finetune':
        model = FinetuneModel
    else:
        model = LithoSimulModel
        
    instance = model(opt)
    print("model [%s] was created" % type(instance).__name__)
    return instance
