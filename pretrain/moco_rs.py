# Satellite contrastive pretraining entry point.
#
# Derived from facebookresearch/moco-v3 (Apache-2.0); see NOTICE.md.
# Mirrors moco_sv.py but uses the satellite-image normalization and datasets.
# This cleaned entry point has no experiment-tracking service dependency and
# contains no embedded credentials.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from functools import partial

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.utils.tensorboard import SummaryWriter

import moco.builder
import moco.loader
import moco.optimizer
import vits

ImageFile.LOAD_TRUNCATED_IMAGES = True


torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)
model_names = ["vit_small", "vit_base", "vit_conv_small", "vit_conv_base"] + torchvision_model_names

parser = argparse.ArgumentParser(description="MoCo-v3 satellite pretraining")
parser.add_argument("data", metavar="DIR", help="path to the training-set pickle")
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument("-j", "--workers", default=32, type=int, metavar="N")
parser.add_argument("--save-folder", default=".", type=str, metavar="PATH")
parser.add_argument("--epochs", default=100, type=int, metavar="N")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
parser.add_argument(
    "-b",
    "--batch-size",
    default=4096,
    type=int,
    metavar="N",
    help="total batch size across all GPUs/nodes",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.6,
    type=float,
    metavar="LR",
    dest="lr",
    help="initial optimizer learning rate (not batch-size rescaled)",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    dest="weight_decay",
)
parser.add_argument("-p", "--print-freq", default=10, type=int, metavar="N")
parser.add_argument("--resume", default="", type=str, metavar="PATH")
parser.add_argument("--pretrained", default="", type=str, metavar="PATH")
parser.add_argument("--world-size", default=-1, type=int)
parser.add_argument("--rank", default=-1, type=int)
parser.add_argument("--dist-url", default="tcp://224.66.41.62:23456", type=str)
parser.add_argument("--dist-backend", default="nccl", type=str)
parser.add_argument("--seed", default=None, type=int)
parser.add_argument("--gpu", default=None, type=int)
parser.add_argument("--multiprocessing-distributed", action="store_true")

# MoCo-specific configuration.
parser.add_argument("--moco-dim", default=256, type=int)
parser.add_argument("--moco-mlp-dim", default=4096, type=int)
parser.add_argument("--moco-m", default=0.99, type=float)
parser.add_argument("--moco-m-cos", action="store_true")
parser.add_argument("--moco-t", default=1.0, type=float)
parser.add_argument("--stop-grad-conv1", action="store_true")
parser.add_argument("--optimizer", default="lars", choices=["lars", "adamw"])
parser.add_argument("--warmup-epochs", default=10, type=int, metavar="N")
parser.add_argument("--crop-min", default=0.2, type=float)


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn(
            "A deterministic seed was requested; CUDNN deterministic mode may slow training."
        )

    if args.gpu is not None:
        warnings.warn("Specifying one GPU disables data parallelism in this entry point.")

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    print("Available GPU:", ngpus_per_node)

    if not torch.cuda.is_available():
        raise RuntimeError("MoCo-v3 pretraining requires CUDA GPUs in this implementation.")

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.multiprocessing_distributed and (args.gpu != 0 or args.rank != 0):
        def print_pass(*_args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print(f"Use GPU: {args.gpu} for training")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        dist.barrier()

    print(f"=> creating model '{args.arch}'")
    if args.arch.startswith("vit"):
        model = moco.builder.MoCo_ViT(
            partial(vits.__dict__[args.arch], stop_grad_conv1=args.stop_grad_conv1),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )
    else:
        model = moco.builder.MoCo_ResNet(
            partial(torchvision_models.__dict__[args.arch], zero_init_residual=True),
            args.moco_dim,
            args.moco_mlp_dim,
            args.moco_t,
        )

    # IMPORTANT: --lr is the actual initial optimizer LR reported in the
    # manuscript. The stock MoCo-v3 launcher scales a base LR by batch_size/256;
    # that rescaling is intentionally not applied here.

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    if args.optimizer == "lars":
        optimizer = moco.optimizer.LARS(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler()
    summary_writer = SummaryWriter() if args.rank == 0 else None

    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"resume checkpoint not found: {args.resume}")
        print(f"=> loading checkpoint '{args.resume}'")
        loc = None if args.gpu is None else f"cuda:{args.gpu}"
        checkpoint = torch.load(args.resume, map_location=loc)
        args.start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"pretrained checkpoint not found: {args.pretrained}")
        print(f"=> loading pretrained model '{args.pretrained}'")
        loc = None if args.gpu is None else f"cuda:{args.gpu}"
        checkpoint = torch.load(args.pretrained, map_location=loc)
        model.load_state_dict(checkpoint["state_dict"])

    cudnn.benchmark = True

    # RS5M-style channel statistics used by the original satellite pipeline.
    normalize = transforms.Normalize(
        mean=[0.406, 0.423, 0.390], std=[0.188, 0.175, 0.185]
    )
    augmentation1 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=1.0),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    augmentation2 = [
        transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.1),
        transforms.RandomApply([moco.loader.Solarize()], p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]

    if "self" in os.path.basename(args.data):
        train_dataset = RSDatasetSelf(
            args.data,
            moco.loader.TwoCropsTransform(
                transforms.Compose(augmentation1), transforms.Compose(augmentation2)
            ),
        )
    else:
        train_dataset = RSDatasetSpatial(args.data, transforms.Compose(augmentation1))

    train_sampler = (
        torch.utils.data.distributed.DistributedSampler(train_dataset)
        if args.distributed
        else None
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, optimizer, scaler, summary_writer, epoch, args)

        if not args.multiprocessing_distributed or args.rank == 0:
            os.makedirs(args.save_folder, exist_ok=True)
            save_path = os.path.join(args.save_folder, f"checkpoint_{epoch}.pth.tar")
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                filename=save_path,
            )

    if args.rank == 0 and summary_writer is not None:
        summary_writer.close()


def train(train_loader, model, optimizer, scaler, summary_writer, epoch, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    learning_rates = AverageMeter("LR", ":.4e")
    losses = AverageMeter("Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()
    end = time.time()
    iters_per_epoch = len(train_loader)
    moco_m = args.moco_m

    for i, (images, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        lr = adjust_learning_rate(optimizer, epoch + i / iters_per_epoch, args)
        learning_rates.update(lr)
        if args.moco_m_cos:
            moco_m = adjust_moco_momentum(epoch + i / iters_per_epoch, args)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast(True):
            loss = model(images[0], images[1], moco_m)

        losses.update(loss.item(), images[0].size(0))
        if summary_writer is not None:
            summary_writer.add_scalar("loss", loss.item(), epoch * iters_per_epoch + i)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


class AverageMeter:
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    """Half-cycle cosine decay after linear warm-up."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.lr * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - args.warmup_epochs)
                / (args.epochs - args.warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def adjust_moco_momentum(epoch, args):
    return 1.0 - 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs)) * (
        1.0 - args.moco_m
    )


class RSDatasetSelf(torch.utils.data.Dataset):
    def __init__(self, file, transform=None):
        df = pd.read_pickle(file)
        self.transform = transform
        self.imgs = df["path"].tolist()

    def __getitem__(self, index):
        path = self.imgs[index]
        name = os.path.basename(path).split(".")[0]
        with open(path, "rb") as handle:
            img = Image.open(handle).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, name

    def __len__(self):
        return len(self.imgs)


class RSDatasetSpatial(torch.utils.data.Dataset):
    def __init__(self, file, transform=None):
        df = pd.read_pickle(file)
        self.transform = transform
        self.imgs1 = df["path1"].tolist()
        self.imgs2 = df["path2"].tolist()

    def __getitem__(self, index):
        path1 = self.imgs1[index]
        path2 = self.imgs2[index]
        name = os.path.basename(path1).split(".")[0]
        images = []
        for path in (path1, path2):
            with open(path, "rb") as handle:
                image = Image.open(handle).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        return images, name

    def __len__(self):
        return len(self.imgs1)


if __name__ == "__main__":
    main()
