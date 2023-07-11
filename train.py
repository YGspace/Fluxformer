import glob
import pathlib

import numpy
# from vivit2_pytorch import ViViT
import pytorchvideo.data
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from vivit2 import ViViT, ClassificationHead

writer = SummaryWriter(filename_suffix="k_means_k=9")

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

# UCF101 101, SSV2 174
dataset_name = "ssv2"

if dataset_name=="HMDB51":
    num_classes=51
    fps = 30
elif dataset_name=="UCF101":
    num_classes = 101
    fps = 25
elif dataset_name=="kinetics400":
    num_classes = 400
    fps = 30
elif dataset_name=="kinetics600":
    num_classes = 600
    fps = 30
elif dataset_name=="ssv2":
    num_classes = 17
    fps = 12


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize_to = 224
num_frames_to_sample = 16
sample_rate = 4

# ucf101 25fps
# hmdb51 30fps
_BATCH_SIZE = 8  # 16
_NUM_WORKERS = 8
clip_duration = num_frames_to_sample * sample_rate / fps

import os


class UCF101DataModule():
    # Training dataset transformations.
    def __init__(self):
        self.dataset_root_path = '/home/vimlab/workspace/datasets/UCF-101/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        video_count_train = len(list(self.dataset_root_path.glob("train/*/*.avi")))
        video_count_val = len(list(self.dataset_root_path.glob("val/*/*.avi")))
        video_count_test = len(list(self.dataset_root_path.glob("test/*/*.avi")))
        video_total = video_count_train + video_count_val + video_count_test
        print(f"Total videos: {video_total}")

        all_video_file_paths = (
                list(self.dataset_root_path.glob("train/*/*.avi"))
                + list(self.dataset_root_path.glob("val/*/*.avi"))
                + list(self.dataset_root_path.glob("test/*/*.avi"))
        )
        print(all_video_file_paths[:5])

        class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
        print(class_labels)

        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}

        print(f"Unique classes: {list(label2id.keys())}.")

    def train_dataloader(self):
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        # Training dataset.
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )

    # Validation and evaluation datasets' transformations.
    def val_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

        # Validation and evaluation datasets.
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )


class kinetics400DataModule():
    # Training dataset transformations.
    def __init__(self):
        self.dataset_root_path = '/home/vimlab/workspace/datasets/kinetics400_5per/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        video_count_train = len(list(self.dataset_root_path.glob("train/*/*.mp4")))
        video_count_val = len(list(self.dataset_root_path.glob("val/*/*.mp4")))
        video_total = video_count_train + video_count_val
        print(f"Total videos: {video_total}")

        all_video_file_paths = (
                list(self.dataset_root_path.glob("train/*/*.mp4"))
                + list(self.dataset_root_path.glob("val/*/*.mp4"))
        )
        print(all_video_file_paths[:5])

        class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
        print(class_labels)

        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}

        print(f"Unique classes: {list(label2id.keys())}.")

    def train_dataloader(self):
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        # Training dataset.
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            
        )

    # Validation and evaluation datasets' transformations.
    def val_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

        # Validation and evaluation datasets.
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            
        )


class kinetics600DataModule():
    # Training dataset transformations.
    def __init__(self):
        self.dataset_root_path = '/home/vimlab/workspace/datasets/kinetics600_5per/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        video_count_train = len(list(self.dataset_root_path.glob("train/*/*.mp4")))
        video_count_val = len(list(self.dataset_root_path.glob("val/*/*.mp4")))
        video_total = video_count_train + video_count_val
        print(f"Total videos: {video_total}")

        all_video_file_paths = (
                list(self.dataset_root_path.glob("train/*/*.mp4"))
                + list(self.dataset_root_path.glob("val/*/*.mp4"))
        )
        print(all_video_file_paths[:5])

        class_labels = sorted({str(path).split("/")[-2] for path in all_video_file_paths})
        print(class_labels)

        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}

        print(f"Unique classes: {list(label2id.keys())}.")

    def train_dataloader(self):
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        # Training dataset.
        train_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "train"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )

    # Validation and evaluation datasets' transformations.
    def val_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

        # Validation and evaluation datasets.
        val_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.Ucf101(
            data_path=os.path.join(self.dataset_root_path, "test"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )


class HMDB51DataModule(pytorch_lightning.LightningDataModule):
    # Training dataset transformations.
    def __init__(self):
        self.dataset_root_path = '/home/vimlab/workspace/datasets/hmdb51/data/'
        self.video_path_prefix = self.dataset_root_path
        self.split_id = 1
        # self.dataset_root_path = '/home/vimlab/workspace/datasets/UCF-101/ii/'
        # self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        # video_count_train = len(list(self.dataset_root_path.glob("train/*/*.avi")))
        # video_count_val = len(list(self.dataset_root_path.glob("val/*/*.avi")))
        # video_count_test = len(list(self.dataset_root_path.glob("test/*/*.avi")))
        all_video_file_paths = glob.glob(self.dataset_root_path + "*")

        class_labels = sorted({str(path).split("/")[-1] for path in all_video_file_paths})

        self.label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in self.label2id.items()}
        print(self.label2id)
        # print(f"Unique classes: {list(label2id.keys())}.")
        self.dataset_root_path = '/home/vimlab/workspace/datasets/hmdb51/testTrainMulti_7030_splits'

    def train_dataloader(self):
        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        # Training dataset.
        train_dataset = pytorchvideo.data.Hmdb51(
            data_path=self.dataset_root_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            decode_audio=False,
            transform=train_transform,
            split_id=self.split_id,
            split_type="train",
            video_path_prefix=self.video_path_prefix,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        ), self.label2id

    # Validation and evaluation datasets' transformations.
    def val_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )

        # Validation and evaluation datasets.
        val_dataset = pytorchvideo.data.Hmdb51(
            data_path=self.dataset_root_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
            split_id=self.split_id,
            split_type="test",
            video_path_prefix=self.video_path_prefix,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.Hmdb51(
            data_path=self.dataset_root_path,
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            decode_audio=False,
            transform=val_transform,
            split_id=self.split_id,
            split_type="test",
            video_path_prefix=self.video_path_prefix,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )


class SSV2DataModule():

    # Dataset configuration
    # _DATA_PATH = dataset_root_path

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """

        train_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        train_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/vimlab/workspace/datasets/ssv2/labels.json",
            video_label_file="/home/vimlab/workspace/datasets/ssv2/train.json",
            video_path_label_file="/home/vimlab/workspace/datasets/ssv2/train.csv",
            video_path_prefix="/home/vimlab/workspace/datasets/ssv2/mp4/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        val_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/vimlab/workspace/datasets/ssv2/labels.json",
            video_label_file="/home/vimlab/workspace/datasets/ssv2/validation.json",
            video_path_label_file="/home/vimlab/workspace/datasets/ssv2/val.csv",
            video_path_prefix="/home/vimlab/workspace/datasets/ssv2/mp4/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(mean, std),
                            Resize((resize_to, resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/vimlab/workspace/datasets/ssv2/labels.json",
            video_label_file="/home/vimlab/workspace/datasets/ssv2/validation.json",
            video_path_label_file="/home/vimlab/workspace/datasets/ssv2/val.csv",
            video_path_prefix="/home/vimlab/workspace/datasets/ssv2/mp4/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
        )


class VideoTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # epoch=30, batch_size=4, num_workers=4, resume=False, resume_from_checkpoint=None, log_interval=30, save_ckpt_freq=20,
        # objective='supervised', eval_metrics='finetune', gpus=-1, root_dir='./', num_class=174, num_samples_per_cls=10000, 
        # img_size=224, num_frames=16, frame_interval=16, multi_crop=False, mixup=False, auto_augment=None, arch='vivit', attention_type='fact_encoder', 
        # pretrain_pth='models/vivit_model.pth', weights_from='imagenet', seed=0, optim_type='sgd', lr_schedule='cosine', lr=7.8125e-05, layer_decay=0.75,
        # min_lr=1e-06, use_fp16=True, weight_decay=0.05, weight_decay_end=0.05, clip_grad=0, warmup_epochs=5)
        # self.model = vit_base_patch16_224()

        self.model = ViViT(
            pretrain_pth="models/vivit_model.pth",
            weights_from="imagenet",
            img_size=224,
            num_frames=16,
            attention_type="fact_encoder")

        self.cls_head = ClassificationHead(
            num_classes, self.model.embed_dims, eval_metrics="finetune")

    def forward(self, x):
        preds = self.model(x)
        results = self.cls_head(preds)
        return results


def cul_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


if __name__ == "__main__":
    epoch = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(device)
    print(device)
    print(device)

    # data_module = SSV2DataModule()
    if dataset_name == "UCF101":
        print(dataset_name)
        data_module = UCF101DataModule()
        train_dataloader = data_module.train_dataloader()
    elif dataset_name == "kinetics400":
        print(dataset_name)
        data_module = kinetics400DataModule()
        train_dataloader = data_module.train_dataloader()
    elif dataset_name == "kinetics600":
        print(dataset_name)
        data_module = kinetics600DataModule()
        train_dataloader = data_module.train_dataloader()
    elif dataset_name== "HMDB51":
        print(dataset_name)
        data_module = HMDB51DataModule()
        train_dataloader,label2id = data_module.train_dataloader()
    elif dataset_name== "ssv2":
        print(dataset_name)
        data_module = SSV2DataModule()
        train_dataloader = data_module.train_dataloader()    
    # train_dataloader= data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()


    # train_top1_acc = Accuracy(task="multiclass",num_classes=num_classes).to(device)
    # train_top5_acc = Accuracy(task="multiclass",top_k=5,num_classes=174).to(device)

    net = VideoTransformer()
    # net.load_state_dict(torch.load("save/k-means/best_50_0.6874.pth"))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    best_model = None
    best_loss = 100000
    best_accuracy = -1
    val_loss = 0
    for epoch in range(epoch):  # loop over the dataset multiple times
        print("-----Epoch : ", epoch + 1)
        net.train()
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):

            inputs = data['video'].to(device)

            ##HMDB51 ##
            if dataset_name == "HMDB51":
                labels = data['label']
                for j in range(len(labels)):
                    labels[j] = label2id[labels[j]]
                labels = torch.Tensor(labels)
                labels = labels.type(torch.LongTensor).to(device)
            else:
                labels = data['label'].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # print statistics

            running_loss += loss.item()  # 500 499
            #if i % 500 == 499:  # print every 2000 mini-batches
        writer.add_scalar("Loss/train_iter", running_loss , (epoch + 1) * i)
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss :.4f}')
        running_loss = 0.0
                # top1_acc = self.train_top1_acc(preds.softmax(dim=-1), labels)
                # acc1, acc5 = cul_accuracy(outputs, labels, topk=(1, 5))
                # since we're not training, we don't need to calculate the gradients for our outputs
        if (epoch + 1) % 5 == 0:
            top1_acc = 0
            top1_acc_list = []
            top5_acc_list = []
            net.eval()
            print("**Run validation**")
            with torch.no_grad():
                for i, data in enumerate(val_dataloader, 0):
                    inputs = data['video'].to(device)
                    if dataset_name == "HMDB51":
                        labels = data['label']
                        for j in range(len(labels)):
                            labels[j] = label2id[labels[j]]
                        labels = torch.Tensor(labels)
                        labels = labels.type(torch.LongTensor).to(device)
                    else:
                        labels = data['label'].to(device)

                    outputs = net(inputs)

                    # top1_acc = train_top1_acc(outputs.softmax(dim=-1), labels).item()
                    top1_acc, top5_acc = cul_accuracy(outputs, labels, topk=(1, 5))
                    top1_acc_list.append(top1_acc.item() / 100)
                    top5_acc_list.append(top5_acc.item() / 100)
                final_top1 = round(numpy.mean(top1_acc_list), 4)
                final_top5 = round(numpy.mean(top5_acc_list), 4)
                print(f'[{epoch + 1}] top1: {final_top1:.3f}')
                print(f'[{epoch + 1}] top5: {final_top5:.3f}')
                writer.add_scalar("top1/epoch", final_top1, (epoch + 1))
                writer.add_scalar("top5/epoch", final_top5, (epoch + 1))

            if best_accuracy < final_top1:
                best_accuracy = final_top1
                best_model = net
                PATH = './save/best_' + str(epoch + 1) + "_" + str(best_accuracy) + '.pth'
                torch.save(net.state_dict(), PATH)
    writer.flush()
    writer.close()

    print('Finished Training')
