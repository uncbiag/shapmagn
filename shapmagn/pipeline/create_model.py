from shapmagn.models.model_opt import DFClassifier
def create_model(opt, devie, gpus):
    """
    create model object
    :param opt: ParameterDict, task setting
    :return: model object
    """
    model_name = opt['model']
    if model_name == 'classifier':
        model = DFClassifier()
    else:
        raise ValueError("Model [%s] not recognized." % model_name)
    model.initialize(opt, devie, gpus)
    print("model [%s] was created" % (model.name()))
    return model
