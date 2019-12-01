import numpy as np
import pickle
import os
from . import data_prams
from .data_prams import train_batch_num
from . import data_augmentation


def get_raw_data(cifar_10_dir):
    """
    cifar_10_dir: cifar-10 dataset dir
    return: train_x, train_y, test_x, test_y, label_names (train_x and test_x are in RGB channel last)
    """
    # load meta data
    meta_data_dir = os.path.join(cifar_10_dir, data_prams.meta_file_name)
    with open(meta_data_dir, "rb") as meta_file:
        meta_dict = pickle.load(meta_file, encoding='bytes')
    batch_size = meta_dict[b'num_cases_per_batch']  # 10000
    data_len = meta_dict[b'num_vis']  # 3072
    label_names = meta_dict[b'label_names']
    label_names = list(map(lambda x: x.decode('utf-8'), label_names))
    meta_dict.clear()

    # get train dataset
    # 5 train batch in total
    train_x = np.zeros([train_batch_num * batch_size, data_len]).astype(np.uint8)
    train_label = np.zeros([train_batch_num * batch_size]).astype(np.uint8)
    for bn in range(train_batch_num):
        train_file_dir = os.path.join(cifar_10_dir, data_prams.train_file_name_prefix.format(bn + 1))
        with open(train_file_dir, "rb") as train_file:
            train_dict = pickle.load(train_file, encoding='bytes')
        print("load " + train_dict[b'batch_label'].decode('utf-8'))
        train_x[batch_size * bn:batch_size * (bn + 1)] = np.array(train_dict[b'data'])
        train_label[batch_size * bn:batch_size * (bn + 1)] = np.array(train_dict[b'labels'])
        train_dict.clear()  # release memory
    train_x = train_x.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])

    # get test dataset
    test_file_dir = os.path.join(cifar_10_dir, data_prams.test_file_name)
    with open(test_file_dir, "rb") as test_file:
        test_dict = pickle.load(test_file, encoding='bytes')
    print("load " + test_dict[b'batch_label'].decode('utf-8'))
    test_x = np.array(test_dict[b'data']).astype(np.uint8)
    test_label = np.array(test_dict[b'labels']).astype(np.uint8)
    test_dict.clear()  # release memory
    test_x = test_x.reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1])

    return train_x, train_label, test_x, test_label, label_names


def data_preprocess(
        train_x, train_label, test_x, test_label, class_num,
        to_BGR, to_channel_first, shuffle=True
):
    """
    to_BGR: to_BGR==True: convert x to BGR mode, or keep x as RGB mode
    to_channel_first: to_channel_first==True: convert x to channel first mode which is recommended for GPU
        computation; to_channel_first==False is recommended fot CPU computation
    return: normalized train_x, test_x normalized with means of train_x and one-hot y and train_x means based on channel
    """
    # shuffle
    if shuffle:
        # shuffle data
        index = np.linspace(0, train_x.shape[0] - 1, train_x.shape[0]).astype(np.int)
        np.random.shuffle(index)
        train_x = train_x[index]
        train_label = train_label[index]

    # to BGR
    if to_BGR:
        train_x = train_x[..., ::-1]
        test_x = test_x[..., ::-1]

    # to channel first
    if to_channel_first:
        train_x = train_x.transpose([0, 3, 1, 2])
        test_x = test_x.transpose([0, 3, 1, 2])

    # convert label to one hot code
    train_y = np.eye(train_label.shape[0], class_num)[train_label].astype(np.float32)
    test_y = np.eye(test_label.shape[0], class_num)[test_label].astype(np.float32)

    return train_x, train_y, test_x, test_y


def import_numpy_data(
        cifar_10_dir, load_dir, reload=False, valid_size=5000,
        to_BGR=True, to_channel_first=False
):
    """
    :param cifar_10_dir:
    :param reload: if reload==True, reload whole dataset and do preprocess, else load preprocessed data from disk
        and all parameters behind will take no effect
    :param valid_size: size of valid set
    :return: data_dict
    """
    data_dict = dict()
    if not reload:
        try:
            print("[info]: loading data")
            data_dict['train_x'] = np.load(os.path.join(load_dir, "train_x.npy"))
            data_dict['train_y'] = np.load(os.path.join(load_dir, "train_y.npy"))
            data_dict['valid_x'] = np.load(os.path.join(load_dir, "valid_x.npy"))
            data_dict['valid_y'] = np.load(os.path.join(load_dir, "valid_y.npy"))
            data_dict['test_x'] = np.load(os.path.join(load_dir, "test_x.npy"))
            data_dict['test_y'] = np.load(os.path.join(load_dir, "test_y.npy"))
            data_dict['mean'] = np.load(os.path.join(load_dir, "mean.npy"))
            data_dict['std'] = np.load(os.path.join(load_dir, "std.npy"))
            data_dict['label_names'] = np.load(os.path.join(load_dir, "label_names.pkl"), allow_pickle=True)
            return data_dict

        except FileNotFoundError:
            print("file not found, reload whole dataset")

    # get raw data
    raw_train_x, raw_train_label, raw_test_x, raw_test_label, label_names = get_raw_data(cifar_10_dir)

    # data preprocess
    train_x, train_y, test_x, test_y = data_preprocess(
        raw_train_x, raw_train_label, raw_test_x, raw_test_label, len(label_names),
        to_BGR=to_BGR,
        to_channel_first=to_channel_first,
        shuffle=True
    )

    # split valid set
    valid_x, train_x = train_x[0:valid_size, ...], train_x[valid_size:, ...]
    valid_y, train_y = train_y[0:valid_size, ...], train_y[valid_size:, ...]

    # data augmentation
    # train_x, train_y = data_augmentation.data_augmentation(
    #     train_x,
    #     train_y,
    #     channel_first=to_channel_first
    # )
    # data normalization
    if to_channel_first:
        mean = np.mean(train_x, axis=(0, 2, 3)).astype(np.float32).reshape([1, 3, 1, 1])
    else:
        mean = np.mean(train_x, axis=(0, 1, 2)).astype(np.float32)

    std = np.std(train_x)
    train_x = (train_x - mean) / std
    valid_x = (valid_x - mean) / std
    test_x = (test_x - mean) / std

    print("[info]: mean: {}, std: {}".format(mean, std))

    # save dataset
    print("writing train_x...")
    np.save(os.path.join(load_dir, "train_x.npy"), train_x)
    print("writing train_y...")
    np.save(os.path.join(load_dir, "train_y.npy"), train_y)
    print("writing valid_x...")
    np.save(os.path.join(load_dir, "valid_x.npy"), valid_x)
    print("writing valid_y...")
    np.save(os.path.join(load_dir, "valid_y.npy"), valid_y)
    print("writing test_x...")
    np.save(os.path.join(load_dir, "test_x.npy"), test_x)
    print("writing test_y...")
    np.save(os.path.join(load_dir, "test_y.npy"), test_y)
    print("writing train_x_mean...")
    np.save(os.path.join(load_dir, "mean.npy"), mean)
    print("writing train_x_std...")
    np.save(os.path.join(load_dir, "std.npy"), std)
    print("writing label_names...")
    with open(os.path.join(load_dir, "label_names.pkl"), "wb") as file:
        pickle.dump(label_names, file)

    data_dict['train_x'] = train_x
    data_dict['train_y'] = train_y
    data_dict['valid_x'] = valid_x
    data_dict['valid_y'] = valid_y
    data_dict['test_x'] = test_x
    data_dict['test_y'] = test_y
    data_dict['mean'] = mean
    data_dict['std'] = std
    data_dict['label_names'] = label_names
    return data_dict


def array_to_cuda_tensor(x):
    import torch
    x = torch.Tensor(x).cuda()
    return x


def import_dataset(load_dir, train_to_cuda=False, test_to_cuda=True):
    from torch.utils import data
    import torch

    data_dict = import_numpy_data(
        cifar_10_dir=None,
        load_dir=load_dir,
        reload=False
    )

    train_x = data_dict['train_x']
    train_y = data_dict['train_y'].argmax(axis=1)
    valid_x = data_dict['valid_x']
    valid_y = data_dict['valid_y'].argmax(axis=1)
    test_x = data_dict['test_x']
    test_y = data_dict['test_y'].argmax(axis=1)

    if train_to_cuda:
        train_x = array_to_cuda_tensor(train_x).float()
        train_y = array_to_cuda_tensor(train_y).long()
    else:
        train_x = torch.Tensor(train_x).float()
        train_y = torch.Tensor(train_y).long()

    if test_to_cuda:
        valid_x = array_to_cuda_tensor(valid_x).float()
        valid_y = array_to_cuda_tensor(valid_y).long()
        test_x = array_to_cuda_tensor(test_x).float()
        test_y = array_to_cuda_tensor(test_y).long()
    else:
        valid_x = torch.Tensor(valid_x).float()
        valid_y = torch.Tensor(valid_y).long()
        test_x = torch.Tensor(test_x).float()
        test_y = torch.Tensor(test_y).long()

    train_set = data.TensorDataset(
        train_x, train_y
    )

    valid_set = data.TensorDataset(
        valid_x, valid_y
    )

    test_set = data.TensorDataset(
        test_x, test_y
    )

    return train_set, valid_set, test_set


def import_test(cifar_10_dir, load_dir, channel_first):
    import cv2
    data_dict = import_numpy_data(cifar_10_dir, load_dir, reload=True, valid_size=5000, to_channel_first=channel_first)

    cv2.namedWindow("test", cv2.WINDOW_NORMAL)
    for i in range(20):
        index = np.random.random_integers(0, data_dict['train_x'].shape[0] - 1)
        x = data_dict['train_x'][index]    # channel last
        x = x*data_dict['std'] + data_dict['mean']
        x[x < 0] = 0
        x[x > 255] = 255
        x = x.astype(np.uint8)

        if channel_first:
            x = x[0]
            x = np.transpose(x, [1, 2, 0])

        cv2.imshow("test", x)
        print(data_dict['label_names'][data_dict['train_y'][index].argmax()])
        cv2.waitKey()


if __name__ == "__main__":
    import os

    home = os.getenv("HOME")
    cifar_data_path = os.path.join(home, "nfs/dataset/cifar/cifar-10-python")
    # save_data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "get_data/data")
    save_data_path = "/media/Data/datasets/cifar/cifar-10-python/data"

    import_test(cifar_data_path, save_data_path, True)


