import os
import shutil
import glob
import string
import zipfile
from pathlib import Path
import tarfile
from shutil import move, rmtree
from typing import Any, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torchvision import datasets
from torchvision.datasets.mnist import read_image_file, read_label_file

from torchvision.datasets.utils import (
    download_url,
    check_integrity,
    verify_str_arg,
    download_and_extract_archive,
    extract_archive
)
import tqdm

class MNIST_RGB(datasets.MNIST):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(MNIST_RGB, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.train = train  # training set or test set

        if self._check_legacy_exist():
            self.data, self.targets = self._load_legacy_data()
            return

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self.data, self.targets = self._load_data()
        self.classes = [i for i in range(10)]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except:
            pass

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTM(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_train.pt.tar.gz',
         '191ed53db9933bd85cc9700558847391'),
        ('https://github.com/liyxi/mnist-m/releases/download/data/mnist_m_test.pt.tar.gz',
         'e11cb4d7fff76d7ec588b1134907db59')
    ]

    training_file = "mnist_m_train.pt"
    test_file = "mnist_m_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'MNIST-M')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SynDigit(torch.utils.data.Dataset):
    resources = [
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_train.pt.gz',
         'd0e99daf379597e57448a89fc37ae5cf'),
        ('https://github.com/liyxi/synthetic-digits/releases/download/data/synth_test.pt.gz',
         '669d94c04d1c91552103e9aded0ee625')
    ]

    training_file = "synth_train.pt"
    test_file = "synth_test.pt"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        root = os.path.join(root, 'SynDigit')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." +
                               " You can use download=True to download it")

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        self.classes = [i for i in range(10)]

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.squeeze().numpy(), mode="RGB")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder, self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder, self.test_file)))

    def download(self):
        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder,
                                         extract_root=self.processed_folder,
                                         filename=filename, md5=md5)

class SVHN(datasets.SVHN):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super(SVHN, self).__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat["X"]
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.targets = loaded_mat["y"].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.targets, self.targets == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))
        self.classes = np.unique(self.targets)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self) -> None:
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

#torchvision 0.21 version
class EMNIST(datasets.MNIST):
    """`EMNIST <https://www.westernsydney.edu.au/bens/home/reproducible_research/emnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``EMNIST/raw/train-images-idx3-ubyte``
            and  ``EMNIST/raw/t10k-images-idx3-ubyte`` exist.
        split (string): The dataset has 6 different splits: ``byclass``, ``bymerge``,
            ``balanced``, ``letters``, ``digits`` and ``mnist``. This argument specifies
            which one to use.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    url = "https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
    md5 = "58c8d27c78d21e728a6bc7b3cc06412e"
    splits = ("byclass", "bymerge", "balanced", "letters", "digits", "mnist")
    # Merged Classes assumes Same structure for both uppercase and lowercase version
    _merged_classes = {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
    _all_classes = set(string.digits + string.ascii_letters)
    classes_split_dict = {
        "byclass": sorted(list(_all_classes)),
        "bymerge": sorted(list(_all_classes - _merged_classes)),
        "balanced": sorted(list(_all_classes - _merged_classes)),
        "letters": ["N/A"] + list(string.ascii_lowercase),
        "digits": list(string.digits),
        "mnist": list(string.digits),
    }

    def __init__(self, root: Union[str, Path], split: str, **kwargs: Any) -> None:
        self.split = verify_str_arg(split, "split", self.splits)
        self.training_file = self._training_file(split)
        self.test_file = self._test_file(split)
        super().__init__(root, **kwargs)
        self.classes = self.classes_split_dict[self.split]

    @staticmethod
    def _training_file(split) -> str:
        return f"training_{split}.pt"

    @staticmethod
    def _test_file(split) -> str:
        return f"test_{split}.pt"

    @property
    def _file_prefix(self) -> str:
        return f"emnist-{self.split}-{'train' if self.train else 'test'}"

    @property
    def images_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-images-idx3-ubyte")

    @property
    def labels_file(self) -> str:
        return os.path.join(self.raw_folder, f"{self._file_prefix}-labels-idx1-ubyte")

    def _load_data(self):
        return read_image_file(self.images_file), read_label_file(self.labels_file)

    def _check_exists(self) -> bool:
        return all(check_integrity(file) for file in (self.images_file, self.labels_file))

    def download(self) -> None:
        """Download the EMNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)

        download_and_extract_archive(self.url, download_root=self.raw_folder, md5=self.md5)
        gzip_folder = os.path.join(self.raw_folder, "gzip")
        for gzip_file in os.listdir(gzip_folder):
            if gzip_file.endswith(".gz"):
                extract_archive(os.path.join(gzip_folder, gzip_file), self.raw_folder)
        shutil.rmtree(gzip_folder)

class EMNIST_RGB(EMNIST):
    def __init__(self, root, split='letters', train=True, transform=None, target_transform=None, download=False,
                 random_seed=42, num_random_classes=26):
        super(EMNIST_RGB, self).__init__(root, split=split, train=train, transform=transform,
                                         target_transform=target_transform, download=download)
        # letters split인 경우 "N/A" (target 0)를 제외한 알파벳(a~z)이 target 1~26로 매핑됨.
        # num_random_classes가 지정되면, valid target (1~26) 중에서 랜덤으로 선택하여 필터링합니다.
        if split == 'letters' and num_random_classes is not None:
            valid_letters = list(string.ascii_lowercase)  # ['a', 'b', ... 'z']
            valid_targets = np.arange(1, 27)  # 1 ~ 26
            if random_seed is not None:
                np.random.seed(random_seed)
            # 랜덤으로 num_random_classes 개를 샘플링 (중복 없이)
            sampled = np.random.choice(valid_targets, size=num_random_classes, replace=False)
            sampled = np.sort(sampled)  # 정렬해서 매핑 순서를 일정하게 유지
            # self.targets는 torch.Tensor임 (타입: torch.long)
            target_np = self.targets.numpy()
            valid_mask = np.isin(target_np, sampled)
            self.data = self.data[valid_mask]
            self.targets = self.targets[valid_mask]
            # 기존 target 값을 0~(num_random_classes-1)로 remap
            mapping = {old: new for new, old in enumerate(sampled)}
            new_targets = [mapping[x.item()] for x in self.targets]
            self.targets = torch.tensor(new_targets, dtype=torch.long)
            # 클래스 이름 업데이트 (예: target 1 -> 'a', target 3 -> 'c' 등)
            self.classes = [valid_letters[t - 1] for t in sorted(sampled)]
        else:
            raise ValueError("split='letters' and num_random_classes must be specified together.")

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except Exception as e:
            print("이미지 변환 중 오류 발생:", e)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class FashionMNIST_RGB(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(FashionMNIST_RGB, self).__init__(root, train=train, transform=transform,
                                                 target_transform=target_transform, download=download)
        self.train = train
        self.classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __getitem__(self, index: int):
        img, target = self.data[index], int(self.targets[index])
        try:
            img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        except Exception as e:
            print("이미지 변환 중 오류 발생:", e)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

class CORe50(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):
        # CORe50 데이터셋은 root/CORe50 폴더 아래에 저장됩니다.
        self.root = os.path.join(os.path.expanduser(root), 'CORe50')
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

        if mode in ['cil', 'joint']:
            self.folder_type = 'domain_merged'
        elif mode in ['dil','vil']:
            self.folder_type = 'domain_separated'
        else:
            raise ValueError('mode should be one of cil, joint, dil, vil')

        self.url = 'http://bias.csr.unibo.it/maltoni/download/core50/core50_128x128.zip'
        self.filename = 'core50_128x128.zip'
        self.extract_folder = os.path.join(self.root, 'core50_128x128')



        if self.train:
            self.dst = os.path.join(self.root, f"train_{self.folder_type}")
        else:
            self.dst = os.path.join(self.root, f"test_{self.folder_type}")

        # 목적 폴더가 없으면 split()을 통해 재구성합니다.
        if not os.path.exists(self.dst):
            if not os.path.exists(self.extract_folder):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from ' + self.url)
                    download_url(self.url, self.root, filename=self.filename)
                    with zipfile.ZipFile(os.path.join(self.root, self.filename), 'r') as zf:
                        for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                            try:
                                zf.extract(member, self.root)
                            except zipfile.error as e:
                                pass
            self.split(self.extract_folder, self.dst)
            rmtree(self.extract_folder)
            
        if self.train:
            if self.folder_type == 'domain_separated':
                self.data = [datasets.ImageFolder(os.path.join(self.dst, s), transform=transform) for s in self.get_train_sessions()]
            else:
                self.data = datasets.ImageFolder(self.dst, transform=transform)
        else:
            self.data = datasets.ImageFolder(self.dst, transform=transform)

    def get_train_sessions(self):
        return ['s1', 's2', 's4', 's5', 's6', 's8', 's9', 's11']

    def get_test_sessions(self):
        return ['s3', 's7', 's10']

    def get_labels(self):
        return [f'o{i}' for i in range(1, 51)]

    def split(self, src, dst):
        """
        src: 압축 해제된 원본 폴더 
        dst: 최종적으로 구성할 폴더 
        mode에 따라 두 가지 방식으로 데이터를 재구성
        """
        os.makedirs(dst, exist_ok=True)
        if self.train:
            # 학습 데이터 재구성
            if self.folder_type == 'domain_separated':
                # 각 세션 폴더(s1, s2, ... 등)를 dst로 이동
                for s in tqdm.tqdm(self.get_train_sessions(), desc='Preprocessing Training'):
                    src_folder = os.path.join(src, s)
                    if os.path.exists(src_folder):
                        move(src_folder, dst)
            else:
                # cil 또는 joint 모드: 모든 세션의 데이터를 클래스별로 통합
                labels = self.get_labels()
                for label in labels:
                    os.makedirs(os.path.join(dst, label), exist_ok=True)
                for s in tqdm.tqdm(self.get_train_sessions(), desc='Preprocessing Training'):
                    session_folder = os.path.join(src, s)
                    if os.path.exists(session_folder):
                        for label in labels:
                            pattern = os.path.join(session_folder, label, '*.png')
                            files = glob.glob(pattern)
                            for file in files:
                                move(file, os.path.join(dst, label))
                        rmtree(session_folder)
        else:
            # 테스트 데이터 재구성: 모든 세션 데이터를 클래스별로 통합
            labels = self.get_labels()
            for label in labels:
                os.makedirs(os.path.join(dst, label), exist_ok=True)
            for s in tqdm.tqdm(self.get_test_sessions(), desc='Preprocessing Testing'):
                session_folder = os.path.join(src, s)
                if os.path.exists(session_folder):
                    for label in labels:
                        pattern = os.path.join(session_folder, label, '*.png')
                        files = glob.glob(pattern)
                        for file in files:
                            move(file, os.path.join(dst, label))
                    rmtree(session_folder)

class DomainNet(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode='cil'):
        root = os.path.join(root, 'DomainNet')   
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform=target_transform
        self.train = train
        self.mode = mode

        if self.mode in ['cil', 'joint']:
            self.folder_type = 'domain_merged'
        else:
            self.folder_type = 'domain_separated'

        
        self.url = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip'
        ]

        self.filename = [
            'clipart.zip',
            'infograph.zip',
            'painting.zip',
            'quickdraw.zip',
            'real.zip',
            'sketch.zip'
        ]

        self.train_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt'
        ]

        for u in self.train_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)
        
        self.test_url_list = [
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
            'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'
        ]

        for u in self.test_url_list:
            filename = u.split('/')[-1]
            if not os.path.isfile(os.path.join(self.root, filename)):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+filename)
                    download_url(u, root, filename=filename)

        self.fpath = [os.path.join(self.root, f) for f in self.filename]

        for i in range(len(self.fpath)):
            if not os.path.isfile(self.fpath[i]):
                if not download:
                    raise RuntimeError('Dataset not found. You can use download=True to download it')
                else:
                    print('Downloading from '+self.url[i])
                    download_url(self.url[i], root, filename=self.filename[i])

        if not os.path.exists(self.root + f"/train_{self.folder_type}") and not os.path.exists(self.root + f"/test_{self.folder_type}"):
            for i in range(len(self.fpath)):
                if not os.path.exists(os.path.join(self.root, self.filename[i][:-4])):
                    with zipfile.ZipFile(os.path.join(self.root, self.filename[i]), 'r') as zf:
                        for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename[i]}'):
                            try:
                                zf.extract(member, root)
                            except zipfile.error as e:
                                pass
            
            self.split()
        
        if self.train:
            fpath = self.root + f"/train_{self.folder_type}"
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)
        else:
            fpath = self.root + f"/test_{self.folder_type}"
            if self.mode not in ['cil', 'joint']:
                self.data = [datasets.ImageFolder(f'{fpath}/{d}', transform=transform) for d in ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']]
            else:
                self.data = datasets.ImageFolder(fpath, transform=transform)

    def split(self):
        train_folder = self.root + f"/train_{self.folder_type}"
        test_folder = self.root + f"/test_{self.folder_type}"
        if os.path.exists(train_folder):
            rmtree(train_folder)
        if os.path.exists(test_folder):
            rmtree(test_folder)
        os.mkdir(train_folder)
        os.mkdir(test_folder)

        if self.folder_type == 'domain_separated':
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(train_folder, path)

                        move(src, dst)
            
            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = os.path.join(test_folder, path)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))
        else:
            for i in tqdm.tqdm(range(len(self.train_url_list)), desc='Preprocessing'):
                train_list = self.train_url_list[i].split('/')[-1]
                
                with open(os.path.join(self.root, train_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])
                        
                        if not os.path.exists(os.path.join(train_folder, dst)):
                            os.makedirs(os.path.join(train_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(train_folder, dst)

                        move(src, dst)

            for i in tqdm.tqdm(range(len(self.test_url_list)), desc='Preprocessing'):
                test_list = self.test_url_list[i].split('/')[-1]

                with open(os.path.join(self.root, test_list), 'r') as f:
                    for line in f.readlines():
                        line = line.replace('\n', '')
                        path, _ = line.split(' ')
                        dst = '/'.join(path.split('/')[1:2])

                        if not os.path.exists(os.path.join(test_folder, dst)):
                            os.makedirs(os.path.join(test_folder, dst))

                        src = os.path.join(self.root, path)
                        dst = '/'.join(path.split('/')[1:])
                        dst = os.path.join(test_folder, dst)

                        move(src, dst)
                rmtree(os.path.join(self.root, test_list.split('_')[0]))

class CLEAR(torch.utils.data.Dataset):
    train_url = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-train-image-only.zip"
    test_url  = "https://huggingface.co/datasets/elvishelvis6/CLEAR-Continual_Learning_Benchmark/resolve/main/clear100-test.zip"
    train_filename = "clear100-train-image-only.zip"
    test_filename  = "clear100-test.zip"
    
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, mode=None, args = None):
        root = os.path.join(root, 'CLEAR_100')
        self.root = os.path.expanduser(root)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.args = args

        if self.mode in ['cil', 'joint']:
            self.folder_type = 'domain_merged'
        elif self.mode in ['dil']:
            self.folder_type = 'domain_separated_11'
        elif self.mode in ['vil']:
            self.folder_type = 'domain_separated_5'
        else:
            raise ValueError('mode should be one of cil, joint, dil, vil')

        if self.train:
            src = os.path.join(self.root, "train_image_only", "labeled_images")
            dst = os.path.join(self.root, f"train_{self.folder_type}")
        else:
            src = os.path.join(self.root, "test",  "labeled_images")
            dst = os.path.join(self.root, f"test_{self.folder_type}")
                
        if not os.path.exists(dst): # 폴더가 없으면 다운로드 및 압축 해제
            if download:
                self.download()
            self.split(src, dst)
            if self.train:
                rmtree(os.path.join(self.root, "train_image_only"))
            else:
                rmtree(os.path.join(self.root, "test"))
        self.fpath = dst
        
        if self.folder_type == 'domain_separated_11':
            domain_list = [str(i) for i in range(0,11)]
            self.data = [datasets.ImageFolder(os.path.join(self.fpath, d), transform=transform) for d in domain_list]
        elif self.folder_type == 'domain_separated_5':
            domain_list = [str(i) for i in range(0,5)]
            self.data = [datasets.ImageFolder(os.path.join(self.fpath, d), transform=transform) for d in domain_list]
        else:
            self.data = datasets.ImageFolder(self.fpath, transform=transform)
    
    def split(self, src, dst):
        """
        src: 원본 데이터가 있는 폴더 (예: .../labeled_images)
        dst: 최종 train 또는 test 폴더 (예: .../OODVIL_train 또는 .../OODVIL_test)
        
        mode에 따라 두 가지로 동작:
        1) mode not in ['cil','joint']: 도메인별로 폴더를 그대로 옮기는데, 
            여기서는 도메인 "1"~"10"을 2개씩 묶어 5개의 도메인으로 재구성.
        2) mode in ['cil','joint']: 모든 도메인의 데이터를 클래스별로 합쳐 단일 폴더 구조로 재구성.
        """

        os.mkdir(dst)
        
        if self.folder_type in ['domain_separated_5','domain_separated_11']:
            if self.folder_type == 'domain_separated_5':
                # 도메인 "1"부터 "10"을 2개씩 묶어서 5개의 그룹으로 만듦
                groups = [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"], ["9", "10"]]
                self.args.num_domains = 5
            elif self.folder_type == 'domain_separated_11':
                groups = [[str(i)] for i in range(0,11)]
                self.args.num_domains = 11
            for i, group in tqdm.tqdm(enumerate(groups, start=0), desc='Preprocessing'):
                new_domain = str(i)
                new_domain_path = os.path.join(dst, new_domain)
                os.mkdir(new_domain_path)
                for d in group:
                    src_d = os.path.join(src, d)
                    if not os.path.exists(src_d):
                        print(f"Warning: 도메인 폴더 {src_d}가 존재하지 않습니다.")
                        continue
                    for class_name in os.listdir(src_d):
                        class_src = os.path.join(src_d, class_name)
                        if os.path.isdir(class_src):
                            class_dst = os.path.join(new_domain_path, class_name)
                            if not os.path.exists(class_dst):
                                os.mkdir(class_dst)
                            for ext in ['*.png', '*.jpg']:
                                files = glob.glob(os.path.join(class_src, ext))
                                for f in files:
                                    move(f, class_dst)
                            rmtree(class_src)
                for d in group:
                    src_d = os.path.join(src, d)
                    if os.path.exists(src_d):
                        rmtree(src_d)
        else:
            domain_list = sorted([d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))])
            sample_domain = os.path.join(src, domain_list[0])
            class_names = sorted([c for c in os.listdir(sample_domain) if os.path.isdir(os.path.join(sample_domain, c))])
            for c in class_names:
                os.mkdir(os.path.join(dst, c))
            exts = ['*.png', '*.jpg']
            for d in tqdm.tqdm(domain_list,desc='Preprocessing'):
                domain_path = os.path.join(src, d)
                for c in os.listdir(domain_path):
                    class_path = os.path.join(domain_path, c)
                    if os.path.isdir(class_path):
                        files = []
                        for ext in exts:
                            files.extend(glob.glob(os.path.join(class_path, ext)))
                        for f in files:
                            move(f, os.path.join(dst, c))
                rmtree(domain_path)
    
    def download(self):
        os.makedirs(self.root, exist_ok=True)
        if self.train:
            train_zip = os.path.join(self.root, self.train_filename)
            if not os.path.exists(train_zip):
                print(f"Downloading train data from {self.train_url} ...")
                download_url(self.train_url, self.root, self.train_filename)
            else:
                print(f"Train zip file {train_zip} already exists.")
            train_extract_dir = os.path.join(self.root, "train_image_only")
            if not os.path.exists(train_extract_dir):
                print(f"Extracting {train_zip} ...")
                extract_archive(train_zip, self.root)
            else:
                print(f"Train extract directory {train_extract_dir} already exists.")
        else:
            test_zip  = os.path.join(self.root, self.test_filename)
            if not os.path.exists(test_zip):
                print(f"Downloading test data from {self.test_url} ...")
                download_url(self.test_url, self.root, self.test_filename)
            else:
                print(f"Test zip file {test_zip} already exists.")   
            test_extract_dir = os.path.join(self.root, "test")
            if not os.path.exists(test_extract_dir):
                print(f"Extracting {test_zip} ...")
                extract_archive(test_zip, self.root)
            else:
                print(f"Test extract directory {test_extract_dir} already exists.")

# ---------------------------------------------------------------------------
# New: KMNIST & QMNIST RGB wrappers (grayscale ➜ RGB)                         
# ---------------------------------------------------------------------------


class KMNIST_RGB(datasets.KMNIST):
    """RGB wrapper around torchvision.datasets.KMNIST (10 classes, 28×28 gray)."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.classes = [i for i in range(10)]  # digits 0-9

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # Grayscale ➜ RGB conversion
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class QMNIST_RGB(datasets.QMNIST):
    """RGB wrapper around torchvision.datasets.QMNIST (digits).

    Args mirror torchvision interface; we simply map to super().__init__.
    """

    def __init__(self, root, what=None, compat=True, train=True, transform=None,
                 target_transform=None, download=False):
        # When 'what' is None, torchvision decides based on 'train'.
        super().__init__(root=root, what=what, compat=compat, train=train,
                         transform=transform, target_transform=target_transform, download=download)
        # QMNIST targets are digits 0-9
        self.classes = [i for i in range(10)]

    def __getitem__(self, index):
        # Fetch raw tensor before any transform and convert to RGB first,
        # mirroring KMNIST_RGB and MNIST_RGB pattern to avoid mode/shape issues.
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

# ---------------------------------------------------------------------------
# Additional dataset wrappers (NotMNIST, PermutedMNIST, Flowers102, etc.)
# ---------------------------------------------------------------------------


class NotMNIST(MNIST_RGB):
    """NotMNIST dataset (A–J glyphs) converted to RGB."""

    url = 'https://github.com/facebookresearch/Adversarial-Continual-Learning/raw/main/data/notMNIST.zip'
    filename = 'notMNIST.zip'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        zip_path = os.path.join(root, self.filename)
        if not os.path.isfile(zip_path):
            if not download:
                raise RuntimeError('Dataset not found. Use download=True to download it.')
            download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'notMNIST')):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error:
                        pass

        fpath = os.path.join(root, 'notMNIST', 'Train' if train else 'Test')

        X, Y = [], []
        folders = os.listdir(fpath)
        for folder in folders:
            folder_path = os.path.join(fpath, folder)
            for ims in os.listdir(folder_path):
                img_path = os.path.join(folder_path, ims)
                try:
                    X.append(np.array(Image.open(img_path).convert('RGB')))
                    Y.append(ord(folder) - 65)  # A–J → 0–9
                except Exception:
                    # skip broken
                    continue
        self.data = np.array(X)
        self.targets = Y

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class PermutedMNIST(datasets.MNIST):
    """MNIST with a fixed random pixel permutation."""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, random_seed=42):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.random_seed = random_seed

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        size = img.size(0)
        np.random.seed(self.random_seed)
        perm = np.random.permutation(size * size)
        img = img.flatten()[perm].reshape(size, size)
        img = Image.fromarray(img.numpy(), mode='L').convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# ----------------------------- ImageNet scale datasets ---------------------

class Flowers102(datasets.Flowers102):
    """RGB Flowers102 wrapper supporting split argument."""

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        from scipy.io import loadmat  # optional dependency
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)

        self._split = verify_str_arg(split, "split", ("train", "val", "test"))
        base = Path(self.root) / "flowers-102"
        images_folder = base / "jpg"

        if download:
            self.download()

        # build index lists
        set_ids = loadmat(base / self._file_dict["setid"][0], squeeze_me=True)
        image_ids = set_ids[self._splits_map[self._split]].tolist()
        labels = loadmat(base / self._file_dict["label"][0], squeeze_me=True)
        image_id_to_label = dict(enumerate(labels["labels"].tolist(), 1))

        self.targets = []
        self._image_files = []
        for image_id in image_ids:
            self.targets.append(image_id_to_label[image_id] - 1)
            self._image_files.append(images_folder / f"image_{image_id:05d}.jpg")

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, index):
        img = Image.open(self._image_files[index]).convert('RGB')
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class StanfordCars(datasets.StanfordCars):
    """Stanford Cars dataset wrapper simplified (train/test split)."""

    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        import scipy.io as sio
        super().__init__(root, transform=transform, target_transform=target_transform, download=download)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        base = Path(self.root) / "stanford_cars"
        devkit = base / "devkit"

        self._annotations_mat_path = (
            base / ("cars_train_annos.mat" if split == 'train' else 'cars_test_annos_withlabels.mat')
        )
        self._images_base_path = base / ("cars_train" if split == 'train' else 'cars_test')

        if download:
            self.download()

        annos = sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        self._samples = [
            (str(self._images_base_path / a["fname"]), a["class"] - 1) for a in annos
        ]
        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        img_path, target = self._samples[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


class CUB200(torch.utils.data.Dataset):
    """CUB-200-2011 dataset (preprocessed into train/test folders)."""

    url = 'https://data.deepai.org/CUB200(2011).zip'
    filename = 'CUB200(2011).zip'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        zip_path = os.path.join(root, self.filename)
        if not os.path.isfile(zip_path):
            if not download:
                raise RuntimeError('Dataset not found. Use download=True to download it')
            download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'CUB_200_2011')):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error:
                        pass

            # inner tar archive
            tar_path = os.path.join(root, 'CUB_200_2011.tgz')
            if os.path.exists(tar_path):
                with tarfile.open(tar_path, 'r') as tf:
                    for member in tqdm.tqdm(tf.getmembers(), desc='Extracting CUB_200_2011.tgz'):
                        try:
                            tf.extract(member, root)
                        except tarfile.TarError:
                            pass
            self._split_into_train_test()

        data_root = os.path.join(root, 'CUB_200_2011', 'train' if train else 'test')
        self.data = datasets.ImageFolder(data_root, transform=transform)

    def _split_into_train_test(self):
        train_folder = os.path.join(self.root, 'CUB_200_2011', 'train')
        test_folder = os.path.join(self.root, 'CUB_200_2011', 'test')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)

        images_file = os.path.join(self.root, 'CUB_200_2011', 'images.txt')
        split_file = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')

        with open(images_file, 'r') as img_f, open(split_file, 'r') as split_f:
            for img_line, split_line in zip(img_f, split_f):
                image_path = img_line.strip().split(' ')[-1]
                is_train = split_line.strip().split(' ')[-1] == '1'
                class_name = image_path.split('/')[0]
                src = os.path.join(self.root, 'CUB_200_2011', 'images', image_path)
                dst_base = train_folder if is_train else test_folder
                dst_dir = os.path.join(dst_base, class_name)
                os.makedirs(dst_dir, exist_ok=True)
                move(src, os.path.join(dst_dir, os.path.basename(image_path)))


class TinyImagenet(torch.utils.data.Dataset):
    """TinyImageNet-200 dataset wrapper (train/test split)."""

    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        zip_path = os.path.join(root, self.filename)
        if not os.path.isfile(zip_path):
            if not download:
                raise RuntimeError('Dataset not found. Use download=True to download it')
            download_url(self.url, root, filename=self.filename)

        if not os.path.exists(os.path.join(root, 'tiny-imagenet-200')):
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for member in tqdm.tqdm(zf.infolist(), desc=f'Extracting {self.filename}'):
                    try:
                        zf.extract(member, root)
                    except zipfile.error:
                        pass
            self._split_val_into_test()

        data_root = os.path.join(root, 'tiny-imagenet-200', 'train' if train else 'test')
        self.data = datasets.ImageFolder(data_root, transform=transform)

    def _split_val_into_test(self):
        val_dir = os.path.join(self.root, 'tiny-imagenet-200', 'val')
        test_dir = os.path.join(self.root, 'tiny-imagenet-200', 'test')
        os.makedirs(test_dir, exist_ok=True)

        val_dict = {}
        with open(os.path.join(val_dir, 'val_annotations.txt'), 'r') as f:
            for line in f.readlines():
                img, cls = line.split('\t')[:2]
                val_dict[img] = cls

        images_dir = os.path.join(val_dir, 'images')
        for img_file in tqdm.tqdm(glob.glob(os.path.join(images_dir, '*')), desc='Moving val images'):
            img_name = os.path.basename(img_file)
            cls = val_dict[img_name]
            dst_cls_dir = os.path.join(test_dir, cls, 'images')
            os.makedirs(dst_cls_dir, exist_ok=True)
            move(img_file, os.path.join(dst_cls_dir, img_name))

        rmtree(val_dir)


class Scene67(torch.utils.data.Dataset):
    """MIT Indoor Scene67 dataset."""

    image_url = 'http://groups.csail.mit.edu/vision/LabelMe/NewImages/indoorCVPR_09.tar'
    train_list_url = 'http://web.mit.edu/torralba/www/TrainImages.txt'
    test_list_url = 'http://web.mit.edu/torralba/www/TestImages.txt'
    image_tar = 'indoorCVPR_09.tar'
    train_txt = 'TrainImages.txt'
    test_txt = 'TestImages.txt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        # download
        for url, fname in [
            (self.image_url, self.image_tar),
            (self.train_list_url, self.train_txt),
            (self.test_list_url, self.test_txt),
        ]:
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                if not download:
                    raise RuntimeError('Dataset not found. Use download=True to download it')
                download_url(url, root, filename=fname)

        if not os.path.exists(os.path.join(root, 'Scene67')):
            # extract images tar
            with tarfile.open(os.path.join(root, self.image_tar), 'r') as tf:
                for member in tqdm.tqdm(tf.getmembers(), desc='Extracting Scene67'):
                    try:
                        tf.extract(member, os.path.join(root, 'Scene67'))
                    except tarfile.TarError:
                        pass
            self._split_train_test()

        data_root = os.path.join(root, 'Scene67', 'train' if train else 'test')
        self.data = datasets.ImageFolder(data_root, transform=transform)

    def _split_train_test(self):
        base = os.path.join(self.root, 'Scene67')
        os.makedirs(os.path.join(base, 'train'), exist_ok=True)
        os.makedirs(os.path.join(base, 'test'), exist_ok=True)
        with open(os.path.join(self.root, self.train_txt), 'r') as f:
            for line in f:
                line = line.strip()
                src = os.path.join(base, 'Images', line)
                dst = os.path.join(base, 'train', line)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                move(src, dst)
        with open(os.path.join(self.root, self.test_txt), 'r') as f:
            for line in f:
                line = line.strip()
                src = os.path.join(base, 'Images', line)
                dst = os.path.join(base, 'test', line)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                move(src, dst)


class Imagenet_R(torch.utils.data.Dataset):
    """ImageNet-Renditions dataset (automatically split 80/20)."""

    url = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'
    filename = 'imagenet-r.tar'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train

        tar_path = os.path.join(root, self.filename)
        if not os.path.isfile(tar_path):
            if not download:
                raise RuntimeError('Dataset not found. Use download=True to download it')
            download_url(self.url, root, filename=self.filename)

        dataset_root = os.path.join(root, 'imagenet-r')
        if not os.path.exists(dataset_root):
            with tarfile.open(tar_path, 'r') as tf:
                for member in tqdm.tqdm(tf.getmembers(), desc='Extracting ImageNet-R'):
                    try:
                        tf.extract(member, root)
                    except tarfile.TarError:
                        pass

        # if not split yet, create train / test split (80/20)
        if not (os.path.exists(os.path.join(dataset_root, 'train')) and os.path.exists(os.path.join(dataset_root, 'test'))):
            full_dataset = datasets.ImageFolder(dataset_root, transform=transform)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_subset, val_subset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

            # save paths
            train_paths = [full_dataset.imgs[i][0] for i in train_subset.indices]
            val_paths = [full_dataset.imgs[i][0] for i in val_subset.indices]

            # reorganize folder structure
            for split_name, paths in [('train', train_paths), ('test', val_paths)]:
                for path in paths:
                    cls = path.split('/')[-2]
                    dst_dir = os.path.join(dataset_root, split_name, cls)
                    os.makedirs(dst_dir, exist_ok=True)
                    move(path, os.path.join(dst_dir, os.path.basename(path)))

            # remove original class folders now empty
            for cls_dir in glob.glob(os.path.join(dataset_root, '*')):
                if os.path.isdir(cls_dir) and cls_dir.split('/')[-1] not in ['train', 'test']:
                    rmtree(cls_dir)

        data_root = os.path.join(dataset_root, 'train' if train else 'test')
        self.data = datasets.ImageFolder(data_root, transform=transform)
