from torch.utils.data import Dataset
from torchvision import transforms
from os.path import join
from PIL import Image
import tqdm
import random


def read_list_from_file(filepath: str) -> list:
    with open(filepath) as f:
        context = f.read()
    context = context.strip().split('\n')
    context = list(x.split(' ')[0] for x in context)
    return context


class WorldIdToLabel:
    def __init__(self, map_filepath):
        with open(map_filepath) as f:
            context = f.read()
        context = context.strip().split('\n')
        context = list(x.strip().split(' ') for x in context)
        self._map = dict()
        for info in context:
            label = int(info[0])
            w_id = info[1]
            self._map[w_id] = label
    
    def __call__(self, w_id):
        return self._map[w_id]


class LabelMap:
    def __init__(self, map_filepath, w_id_to_label):
        with open(map_filepath) as f:
            context = f.read()
        context = context.strip().split('\n')
        context = list(x.strip().split(' ') for x in context)
        
        self._w_id_dict = dict()
        self._label_dict = dict()
        self._name_dict = dict()
        self._imagenet_label_dict = dict()

        for info in context:
            w_id = info[0]
            label = w_id_to_label(w_id)
            imagenet_label = info[1]
            name = info[2]
            self._w_id_dict[w_id] = dict(label=label, name=name)
            self._label_dict[label] = dict(w_id=w_id, name=name)
            self._name_dict[name] = dict(w_id=w_id, label=label)
            self._imagenet_label_dict[imagenet_label] = label

    def w_id_to_label(self, w_id):
        return self._w_id_dict[w_id]["label"]
        
    def w_id_to_name(self, w_id):
        return self._w_id_dict[w_id]["name"]

    def label_to_w_id(self, label):
        return self._label_dict[label]["w_id"]

    def label_to_name(self, label):
        return self._label_dict[label]["name"]
    
    def name_to_w_id(self, name):
        return self._name_dict[name]["w_id"]
    
    def name_to_label(self, name):
        return self._name_dict[name]["label"]

    def imagenet_id_to_label(self, imagenet_id):
        return self._imagenet_label_dict[imagenet_id]


class ImagenetDataset(Dataset):
    def __init__(self, imagenet_root: str, key: str, train_use_percentage=100.0):
        """Args:
        imagenet_root: path to ILSVRC
        key: key of dataset, can be any of {"train", "val"}
        """
        self._root = imagenet_root
        self._key = key
        if key == "train":
            self._img_list = read_list_from_file(join(
                imagenet_root, 
                "ImageSets/CLS-LOC/train_cls.txt"
            ))
            k = int(len(self._img_list)*train_use_percentage/100)
            self._img_list = random.sample(self._img_list, k)
            print("[info] use {}% training examples, total {}.".format(
                train_use_percentage, 
                len(self._img_list))
            )
            
        elif key == "val":
            self._img_list = read_list_from_file(join(
                imagenet_root, 
                "ImageSets/CLS-LOC/val.txt"
            ))
        else:
            raise Exception("Key Error")
        
        self._len = len(self._img_list)
        self._label = list()
        self._word_id_to_label = WorldIdToLabel(join(
            self._root, 
            "devkit/data/imagenet_class_index.txt"
        ))
        self._label_map = LabelMap(
            map_filepath=join(self._root, "devkit/data/map_clsloc.txt"), 
            w_id_to_label=self._word_id_to_label
        )
        self._init_label()
        normalize_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        if self._key == "train":
            self._transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_transform
            ])
        elif self._key == "val":
            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize_transform
            ])
    
    def _init_label(self):
        print("[info] generating {} label list".format(self._key))
        if self._key == "train":
            for name in tqdm.tqdm(self._img_list):
                w_id = name.split('/')[0]
                self._label.append(self._label_map.w_id_to_label(w_id))
        elif self._key == "val":
            val_label_filepath = join(
                self._root, 
                "devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"
            )
            self._label = read_list_from_file(val_label_filepath)
            self._label = list(self._label_map.imagenet_id_to_label(x) for x in self._label)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        img_filepath = join(
            self._root, 
            "Data/CLS-LOC/{}/{}.JPEG".format(self._key, self._img_list[index])
        )

        with open(img_filepath, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
            
        img = self._transform(img)
        return img, self._label[index]

    @property
    def label_map(self) -> LabelMap:
        return self._label_map


if __name__ == "__main__":
    import random
    from torch.utils.data import DataLoader
    dataset = ImagenetDataset("/nfs2/zhfeing/dataset/ILSVRC", "train")
    data_loader = DataLoader(
        dataset=dataset, 
        batch_size=32, 
        num_workers=0, 
        pin_memory=True, 
        shuffle=True
    )
    for x, y in data_loader:
        print(x, y)
        input()
    # for i in range(len(dataset)):
    #     img, label = dataset[random.randint(0, len(dataset) - 1)]
    #     print(
    #         img.shape, 
    #         label, 
    #         dataset.label_map.label_to_name(label), 
    #         dataset.label_map.label_to_w_id(label)
    #     )
    #     print(img)
    #     input()
