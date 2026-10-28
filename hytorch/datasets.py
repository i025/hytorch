import numpy as np
import gzip
import matplotlib.pyplot as plt
from hytorch.utils import get_file
from hytorch.transforms import (
    Compose, Flatten, ToFloat, Normalize,
    ToArray, Pad, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip,
    RandomShift, RandomRotate90,
    RandomBrightness, RandomContrast,
    RandomGaussianNoise, RandomErasing, Standardize
)
import os
import urllib.request


class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform    # 数据预处理
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x      # 匿名函数，传入x，输出x
        if self.target_transform is None:
            self.target_transform = lambda x: x

        self.data = None
        self.label = None
        self.prepare()

    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:     # 无监督学习
            return self.target_transform(self.data[index]), None
        else:
            return self.transform(self.data[index]), self.target_transform(self.label[index])

    def __len__(self):
        return len(self.data)

    def prepare(self):
        pass


def get_spiral(train=True):    # 螺旋数据集
    seed = 1984 if train else 2020
    np.random.seed(seed=seed)

    num_data, num_class, input_dim = 100, 3, 2
    data_size = num_class * num_data
    x = np.zeros((data_size, input_dim), dtype=np.float32)
    t = np.zeros(data_size, dtype=int)

    for j in range(num_class):
        for i in range(num_data):
            rate = i / num_data
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2
            ix = num_data * j + i
            x[ix] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()
            t[ix] = j
    # Shuffle
    indices = np.random.permutation(num_data * num_class)
    x = x[indices]
    t = t[indices]
    return x, t


class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)


class MNIST(Dataset):

    def __init__(self, train=True, target_transform=None, augment=False):
        if train and augment:
            # 训练集：带数据增强
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Pad(2),
                RandomCrop((28, 28)),
                RandomShift((2, 2)),
                RandomRotate90(p=0.1),
                RandomBrightness(0.2),
                RandomContrast(0.2),
                RandomGaussianNoise(0.03),
                RandomErasing(erase_prob=0.5),
                Flatten(),   # 如果是 MLP 需要；CNN 就不要加
            ])
        else:
            # 测试集：只做确定性预处理
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Flatten(),
            ])

        super().__init__(train, transform, target_transform)

    def prepare(self):
        url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
        train_files = {
            "target": "train-images-idx3-ubyte.gz",
            "label": "train-labels-idx1-ubyte.gz",
        }
        test_files = {
            "target": "t10k-images-idx3-ubyte.gz",
            "label": "t10k-labels-idx1-ubyte.gz",
        }

        files = train_files if self.train else test_files
        data_path = get_file(url + files["target"])
        label_path = get_file(url + files["label"])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def _load_label(self, filepath):
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    # def prepare(self):
    #     url = "https://ossci-datasets.s3.amazonaws.com/mnist/"   # 将github的网址换成了其他源
    #     train_files = {
    #         "target": "train-images-idx3-ubyte.gz",
    #         "label": "train-labels-idx1-ubyte.gz",
    #     }
    #     test_files = {
    #         "target": "t10k-images-idx3-ubyte.gz",
    #         "label": "t10k-labels-idx1-ubyte.gz",
    #     }
    #
    #     files = train_files if self.train else test_files
    #     data_path = get_file(url + files["target"])
    #     label_path = get_file(url + files["label"])
    #
    #     self.data = self._load_data(data_path)
    #     self.label = self._load_label(label_path)
    #
    # def _load_label(self, filepath):
    #     with gzip.open(filepath, "rb") as f:
    #         labels = np.frombuffer(f.read(), np.uint8, offset=8)
    #     return labels
    #
    # def _load_data(self, filepath):
    #     with gzip.open(filepath, "rb") as f:
    #         data = np.frombuffer(f.read(), np.uint8, offset=16)
    #     data = data.reshape(-1, 1, 28, 28)
    #     return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H: (r + 1) * H, c * W: (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

    @staticmethod
    def labels():
        return {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "5",
            6: "6",
            7: "7",
            8: "8",
            9: "9",
        }


class FashionMNIST(Dataset):
    # 类变量，所有实例共享
    _default_data_dir = None

    def __init__(self, train=True, target_transform=None, augment=False, data_dir=None):
        # 如果提供了data_dir，更新默认值
        if data_dir is not None:
            FashionMNIST._default_data_dir = data_dir

        # 使用默认数据目录（如果已设置）
        self.data_dir = FashionMNIST._default_data_dir

        # 设置数据增强和预处理
        if train and augment:
            # 训练集：带数据增强
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Pad(2),
                RandomCrop((28, 28)),
                RandomShift((2, 2)),
                RandomRotate90(p=0.1),
                RandomBrightness(0.2),
                RandomContrast(0.2),
                RandomGaussianNoise(0.03),
                RandomErasing(erase_prob=0.5),
                # Flatten(),  # 如果是 MLP 需要；CNN 就不要加
            ])
        else:
            # 测试集：只做确定性预处理
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                # Flatten(),
            ])

        super().__init__(train, transform, target_transform)

    def prepare(self):
        # FashionMNIST 数据源
        url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
        train_files = {
            "target": "train-images-idx3-ubyte.gz",
            "label": "train-labels-idx1-ubyte.gz",
        }
        test_files = {
            "target": "t10k-images-idx3-ubyte.gz",
            "label": "t10k-labels-idx1-ubyte.gz",
        }

        files = train_files if self.train else test_files

        # 如果有数据目录，使用自定义的get_file版本
        if self.data_dir:
            data_path = self.get_file_custom(url + files["target"], self.data_dir)
            label_path = self.get_file_custom(url + files["label"], self.data_dir)
        else:
            # 使用原始的get_file函数
            data_path = get_file(url + files["target"])
            label_path = get_file(url + files["label"])

        self.data = self._load_data(data_path)
        self.label = self._load_label(label_path)

    def get_file_custom(self, url, download_dir):
        """自定义下载函数，指定下载目录"""

        os.makedirs(download_dir, exist_ok=True)
        filename = os.path.basename(url)
        filepath = os.path.join(download_dir, filename)

        if not os.path.exists(filepath):
            print(f"下载 {filename} 到 {filepath}")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"文件已存在: {filepath}")

        return filepath

    def _load_label(self, filepath):
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels

    def _load_data(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                img[r * H: (r + 1) * H, c * W: (c + 1) * W] = self.data[
                    np.random.randint(0, len(self.data) - 1)
                ].reshape(H, W)
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.show()

    @staticmethod
    def labels():
        return {
            0: "T-shirt/top",
            1: "Trouser",
            2: "Pullover",
            3: "Dress",
            4: "Coat",
            5: "Sandal",
            6: "Shirt",
            7: "Sneaker",
            8: "Bag",
            9: "Ankle boot",
        }

    @classmethod
    def set_default_data_dir(cls, data_dir):
        """类方法，设置默认数据目录"""
        cls._default_data_dir = data_dir
        print(f"已设置默认数据目录: {data_dir}")


try:
    import scipy.io
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("警告: 未找到scipy，无法加载SVHN数据集")

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("警告: 未找到matplotlib，无法显示图像")


class SVHN(Dataset):
    _default_data_dir = None

    def __init__(self, train=True, target_transform=None, augment=False, data_dir=None):
        if not SCIPY_AVAILABLE:
            raise ImportError("需要安装scipy来加载SVHN数据集。请运行: pip install scipy")

        if data_dir is not None:
            SVHN._default_data_dir = data_dir

        self.data_dir = SVHN._default_data_dir

        # SVHN是彩色图像，不需要Flatten
        if train and augment:
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Standardize(mean=[0.4377, 0.4438, 0.4728],
                            std=[0.1980, 0.2010, 0.1970]),
                RandomCrop((32, 32)),
                RandomShift((2, 2)),
                RandomRotate90(p=0.1),
                RandomBrightness(0.2),
                RandomContrast(0.2),
                RandomGaussianNoise(0.03),
                RandomErasing(erase_prob=0.3),
            ])
        else:
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Standardize(mean=[0.4377, 0.4438, 0.4728],
                            std=[0.1980, 0.2010, 0.1970]),

            ])

        super().__init__(train, transform, target_transform)

    def prepare(self):
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy未安装，无法加载SVHN数据集")

        # SVHN 数据源
        if self.train:
            url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        else:
            url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"

        # 使用统一的get_file函数下载文件
        file_path = get_file(url, cache_dir=self.data_dir)

        # 使用scipy.io加载.mat文件
        try:
            mat_data = scipy.io.loadmat(file_path)
        except Exception as e:
            print(f"加载.mat文件失败: {e}")
            print("文件可能已损坏，尝试重新下载...")
            # 删除损坏的文件并重新下载
            if os.path.exists(file_path):
                os.remove(file_path)
            file_path = get_file(url)
            mat_data = scipy.io.loadmat(file_path)

        # SVHN数据格式: (32, 32, 3, N) -> 转换为 (N, 3, 32, 32)
        data = mat_data['X']
        labels = mat_data['y']

        # 转换数据格式
        data = np.transpose(data, (3, 2, 0, 1))

        # 标签处理: SVHN标签是1-10，其中10表示数字0
        labels = labels.flatten()
        labels[labels == 10] = 0  # 将10转换为0

        self.data = data
        self.label = labels

    def show(self, row=5, col=5, save_path=None):
        if not MATPLOTLIB_AVAILABLE:
            print("无法显示图像: matplotlib未安装")
            return

        H, W = 32, 32
        img = np.zeros((H * row, W * col, 3))
        for r in range(row):
            for c in range(col):
                idx = np.random.randint(0, len(self.data) - 1)
                # 转换格式: (3, 32, 32) -> (32, 32, 3)
                sample_img = np.transpose(self.data[idx], (1, 2, 0))
                img[r * H: (r + 1) * H, c * W: (c + 1) * W] = sample_img

        plt.figure(figsize=(10, 10))
        plt.imshow(img.astype(np.uint8))
        plt.axis("off")
        plt.title("SVHN Dataset Samples")

        if save_path:
            plt.savefig(save_path)
            print(f"图像已保存到: {save_path}")
        else:
            try:
                plt.show()
            except Exception as e:
                print(f"显示图像时出错: {e}")
                plt.savefig("svhn_sample.png")
                print("图像已保存到: svhn_sample.png")

        plt.close()

    @staticmethod
    def labels():
        return {
            0: "0", 1: "1", 2: "2", 3: "3", 4: "4",
            5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
        }

    @classmethod
    def set_default_data_dir(cls, data_dir):
        cls._default_data_dir = data_dir
        print(f"已设置默认数据目录: {data_dir}")


class EMNIST(Dataset):
    """
    EMNIST: 使用 EMNIST Balanced (47 类) 作为 28x28 灰度扩展数据集。
    - 文件格式：与 MNIST 相同的 IDX .gz
    - 直链写死：HuggingFace 镜像的 .gz 文件（emnist-balanced）
    - data_dir: 外部可传；若传入则下载/缓存到该目录；否则走默认 ~/.hytorch
    - 输出形状：self.data -> (N, 1, 28, 28), self.label -> (N,)
    """
    _default_data_dir = None

    def __init__(self, train=True, target_transform=None, augment=False, data_dir=None):
        if data_dir is not None:
            EMNIST._default_data_dir = data_dir
        self.data_dir = EMNIST._default_data_dir

        # 与 MNIST 风格一致：灰度+可选增强（注意：EMNIST是字符集，增强要保守）
        if train and augment:
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Standardize(mean=[0.286], std=[0.353]),
                Pad(2),
                RandomCrop((28, 28)),
                RandomShift((2, 2)),
                RandomRotate90(p=0.05),
                RandomBrightness(0.1),
                RandomContrast(0.1),
                RandomGaussianNoise(0.01),
                RandomErasing(erase_prob=0.2),
                # 不做 Flatten，便于 CNN
            ])
        else:
            transform = Compose([
                ToArray(np.float32),
                Normalize(0.0, 255.0),
                Standardize(mean=[0.286], std=[0.353]),
            ])

        super().__init__(train, transform, target_transform)

    def prepare(self):
        #def prepare(self):
        # —— 镜像源直链（Hugging Face，.gz 文件）——
        if self.train:
            img_url = "https://huggingface.co/datasets/Royc30ne/emnist-balanced/resolve/main/emnist-balanced-train-images-idx3-ubyte.gz"
            lbl_url = "https://huggingface.co/datasets/Royc30ne/emnist-balanced/resolve/main/emnist-balanced-train-labels-idx1-ubyte.gz"
        else:
            img_url = "https://huggingface.co/datasets/Royc30ne/emnist-balanced/resolve/main/emnist-balanced-test-images-idx3-ubyte.gz"
            lbl_url = "https://huggingface.co/datasets/Royc30ne/emnist-balanced/resolve/main/emnist-balanced-test-labels-idx1-ubyte.gz"

        # —— 本地优先，其次再下载到 data_dir（若你断网，可手动把文件放到 data_dir 即可）——
        base_dir = self.data_dir or os.path.join(os.path.expanduser('~'), '.hytorch')
        os.makedirs(base_dir, exist_ok=True)
        img_local = os.path.join(base_dir, os.path.basename(img_url))
        lbl_local = os.path.join(base_dir, os.path.basename(lbl_url))

        if os.path.exists(img_local) and os.path.exists(lbl_local):
            img_path, lbl_path = img_local, lbl_local
        else:
            img_path = get_file(img_url, cache_dir=base_dir)
            lbl_path = get_file(lbl_url, cache_dir=base_dir)

        # 读取 .gz（IDX） -> (N,1,28,28) & (N,)
        self.data = self._load_data(img_path)
        self.label = self._load_label(lbl_path)


    def _load_label(self, filepath):
        with gzip.open(filepath, "rb") as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)
        return labels.astype(np.int64)

    def _load_data(self, filepath):
        with gzip.open(filepath, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 28, 28)  # (N,28,28)

        # ⚠️ EMNIST 系列常见需要“转置/旋转”才能与 MNIST 方向一致。
        # 简化处理：先做转置 (H,W) -> (W,H)。如仍觉方向不对，可再按需 flip。
        data = np.transpose(data, (0, 2, 1))
        # 若你希望严格对齐 torchvision 的处理，可再做一次上下翻转：
        # data = data[:, ::-1, :]

        data = data[:, None, :, :]       # -> (N,1,28,28)
        return data

    def show(self, row=10, col=10):
        H, W = 28, 28
        img = np.zeros((H * row, W * col))
        for r in range(row):
            for c in range(col):
                idx = np.random.randint(0, len(self.data) - 1)
                img[r*H:(r+1)*H, c*W:(c+1)*W] = self.data[idx].reshape(1, H, W)[0]
        plt.imshow(img, cmap="gray", interpolation="nearest")
        plt.axis("off")
        plt.title("EMNIST (EMNIST Balanced, 47 classes)")
        plt.show()

    @staticmethod
    def labels():
        # 47 类占位：0..46（EMNIST Balanced 的字符集合可参考官方 mapping 文件）
        return {i: str(i) for i in range(47)}

    @classmethod
    def set_default_data_dir(cls, data_dir):
        cls._default_data_dir = data_dir
        print(f"已设置默认数据目录: {data_dir}")

