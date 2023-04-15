import torch


class FakeDataset(torch.utils.data.Dataset):
    def __init__(self, img_size=32, num_classes=10, use_fp16=False):
        self.img_size = img_size  # 图像大小
        self.num_classes = num_classes  # 标签数目
        self.use_fp16 = use_fp16 # 是否使用半精度

    def __len__(self):
        return 10000000  # 数据集大小为无限大

    def __getitem__(self, index):
        # 生成随机数据
        data = torch.randn(3, self.img_size, self.img_size)

        # 生成随机标签
        target = torch.randint(low=0, high=self.num_classes, size=(1,)).item()

        return data, target
