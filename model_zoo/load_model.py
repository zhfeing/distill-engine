import os
import torch


def load_model(version, new_model, just_weights, retrain=False, to_cuda=True, *args):
    """
    :param version: model version
    :param new_model: method for call to get a new model e.g. my_ResNet.my_ResNet
    :param retrain: True: load new model
    :param to_cuda: put model to cuda if True
    :return:
    """
    create_new_model = False
    model = new_model(*args)
    if to_cuda:
        location = None
    else:
        location = 'cpu'
    # load model
    if not retrain:
        try:
            if just_weights:
                model.load_state_dict(torch.load(
                    os.path.join("model_zoo/model", "model_weights_{}.pkl".format(version)),
                    map_location=location
                ))
            else:
                model = torch.load(
                    os.path.join("model_zoo/model", "model_{}.pkl".format(version)),
                    map_location=location
                )

            print("[info]: load model done.")
        except FileNotFoundError:
            print("[info]: load model file failed, create model")
            create_new_model = True
    else:
        print("[info]: retrain, creating model")
        create_new_model = True

    if to_cuda:
        model = model.cuda()
    return model, create_new_model


def save_model(version, model):
    torch.save(model, os.path.join("model_zoo/model", "model_{}.pkl".format(version)))


def print_parameters(model):
    params = list(model.parameters())
    k = 0
    for i in params:
        a = 1
        for j in i.size():
            a *= j
        k = k + a
        # print(l)
    k = format(k, ',')
    print("total parameters: " + k)
