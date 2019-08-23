import torch
import numpy as np
import cv2
import random


def random_crop_img(img, padding_size):
    img_shape = img.shape
    img = cv2.copyMakeBorder(
        src=img,
        top=padding_size, bottom=padding_size, left=padding_size, right=padding_size,
        borderType=cv2.BORDER_REPLICATE
    )
    o_x = np.random.randint(0, 2 * padding_size)
    o_y = np.random.randint(0, 2 * padding_size)
    img = img[o_y:o_y + img_shape[0], o_x:o_x + img_shape[1]]
    return img


def numpy_data_augmentation(train_x, train_y, channel_first, flip_pr=0.5, padding_size=4, noise_std=1.414):
    # padding and randomly flip
    flipped_img_list = []
    flipped_label_list = []

    if channel_first:   # convert to channel last
        train_x = np.transpose(train_x, axes=[0, 2, 3, 1])

    for i in range(train_x.shape[0]):
        img = train_x[i]
        # randomly flip img
        flip = np.random.binomial(1, flip_pr)
        if flip == 1:       # flip this img
            flipped_img = cv2.flip(img, 1)
            flipped_img = random_crop_img(flipped_img, padding_size)
            flipped_img_list.append(flipped_img)
            flipped_label_list.append(train_y[i])

        img = random_crop_img(img, padding_size)
        # write back
        train_x[i] = img

        # test
        # cv2.imshow("img", img)
        # if flip:
        #     cv2.imshow("flipped", flipped_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    flipped_img_list = np.array(flipped_img_list)
    flipped_label_list = np.array(flipped_label_list)

    # append flipped imgs
    train_x = np.append(train_x, flipped_img_list, axis=0)
    train_y = np.append(train_y, flipped_label_list, axis=0)

    print("[info]: append {} images".format(flipped_img_list.shape[0]))

    # add noise
    noise = np.random.randn(*train_x.shape).astype(np.float32)*noise_std
    print("[info]: noise average: {:.4f}, noise var: {:.4f}, noise max: {:.4f}".
          format(np.mean(noise), np.var(noise), np.max(noise)))
    train_x = train_x.astype(np.float32) + noise

    # limiting
    train_x[train_x < 0] = 0
    train_x[train_x > 255] = 255
    train_x = train_x.astype(np.uint8)

    if channel_first:   # convert to channel first
        train_x = np.transpose(train_x, axes=[0, 3, 1, 2])

    return train_x, train_y


def tensor_data_argumentation(x, flip_pr=0.5, padding_size=4, max_rotate_angle=30, max_scale=1.4):
    x = x.detach().numpy()
    x = np.transpose(x, axes=[0, 2, 3, 1])
    for i in range(x.shape[0]):
        img = x[i]
        # randomly flip img
        flip = np.random.binomial(1, flip_pr)
        if flip == 1:       # flip this img
            img = cv2.flip(img, 1)

        # randomly crop img
        img = random_crop_img(img, padding_size)

        # randomly rotate and scale img
        # set parameter
        angle = 2*max_rotate_angle*random.random() - max_rotate_angle   # [-a, a]
        scale = (max_scale - 1.0/max_scale)*random.random() + 1.0/max_scale
        # transform matrix
        rotate_matrix = cv2.getRotationMatrix2D(
            center=(img.shape[1] / 2, img.shape[0] / 2),
            angle=angle,
            scale=scale
        )
        img = cv2.warpAffine(src=img, M=rotate_matrix, dsize=(img.shape[0], img.shape[1]))

        # write back
        x[i] = img

    x = np.transpose(x, axes=[0, 3, 1, 2])
    x = torch.Tensor(x)
    return x

