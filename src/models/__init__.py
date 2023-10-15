from . import ganbert
from . import basic

def get_model(train_set, exp_dict):
    if exp_dict["model"] == "ganbert":
        return ganbert.GanBert(train_set, exp_dict)
    elif exp_dict["model"] == "basic":
        return basic.Basic(train_set, exp_dict)
    else:
        raise ValueError(f'{exp_dict["model"]} does not exist')
