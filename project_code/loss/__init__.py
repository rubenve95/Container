import logging
import functools
import torch

from project_code.loss.loss import (
    cross_entropy2d,
    bootstrapped_cross_entropy2d,
    multi_scale_cross_entropy2d,
)


#logger = logging.getLogger("ptsemseg")

key2loss = {
    "cross_entropy2d": cross_entropy2d,
    "bootstrapped_cross_entropy": bootstrapped_cross_entropy2d,
    "multi_scale_cross_entropy": multi_scale_cross_entropy2d,
}


def get_loss_function(loss_name='cross_entropy2d'):

   # if loss_name is None:
        #logger.info("Using default cross entropy loss")
       # return cross_entropy2d

    if loss_name == "cross_entropy":
        #logger.info("Using default cross entropy loss")
        return torch.nn.CrossEntropyLoss()

    else:        
    #     loss_params = {k: v for k, v in loss_dict.items() if k != "name"}

    #     if loss_name not in key2loss:
    #         raise NotImplementedError("Loss {} not implemented".format(loss_name))

    #     logger.info("Using {} with {} params".format(loss_name, loss_params))
        return functools.partial(key2loss[loss_name])#, **loss_params)
