from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop

def get_optimizer(options, model_params):
    opt_name = options['training']['optimizer']['name']
    if opt_name=='sgd':
        return SGD(model_params, lr=options['training']['optimizer']['lr'],
                    momentum=options['training']['optimizer']['momentum'],
                    weight_decay=options['training']['optimizer']['weight_decay'], 
                    nesterov=options['training']['optimizer']['nesterov'])
    else:
        raise NotImplementedError("Optimizer {} not implemented".format(opt_name))