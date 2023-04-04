
import torch.distributed as dist
import torchvision
import torchvision.transforms as transforms


def cifar_trainset(args, dl_path='/home/xihe/datasets/cifar10'):
    local_rank = args.local_rank
    img_size = args.img_size
    transform = transforms.Compose([
        transforms.Resize(int(img_size/0.875)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ensure only one rank downloads.
    # Note: if the download path is not on a shared filesytem, remove the semaphore
    # and switch to args.local_rank
    dist.barrier()
    if local_rank != 0:
        dist.barrier()
    trainset = torchvision.datasets.CIFAR10(root=dl_path,
                                            train=True,
                                            download=True,
                                            transform=transform)
    if local_rank == 0:
        dist.barrier()
    return trainset

