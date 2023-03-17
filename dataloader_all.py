import glob
import pathlib
# from vivit2_pytorch import ViViT

import torch
import pytorchvideo.data
import pytorch_lightning
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import os
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


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
resize_to = 224
num_frames_to_sample = 16
sample_rate = 4

# ucf101 25fps
# hmdb51 30fps
_BATCH_SIZE = 4  # 16
_NUM_WORKERS = 8


def make_dataloader(name):
    dataset_name = name
    num_classes = 0
    if dataset_name == "HMDB51":
        num_classes = 51
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps
        print(dataset_name)
        data_module = HMDB51DataModule(clip_duration)
        train_dataloader,label2id = data_module.train_dataloader()
    elif dataset_name == "UCF101":
        num_classes = 101
        fps = 25
        clip_duration = num_frames_to_sample * sample_rate / fps
        print(dataset_name)
        data_module = UCF101DataModule(clip_duration)
        train_dataloader = data_module.train_dataloader()
    elif dataset_name == "kinetics400":
        num_classes = 400
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps
        print(dataset_name)
        data_module = kinetics400DataModule(clip_duration)
        train_dataloader = data_module.train_dataloader()
    elif dataset_name == "kinetics600":
        num_classes = 600
        fps = 30
        clip_duration = num_frames_to_sample * sample_rate / fps
        print(dataset_name)
        data_module = kinetics600DataModule(clip_duration)
        train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    return train_dataloader,val_dataloader,num_classes

class UCF101DataModule(pytorch_lightning.LightningDataModule):
    # Training dataset transformations.
    def __init__(self,clip_duration):
        self.dataset_root_path = '/home/hong/workspace/datasets/UCF-101/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        video_count_train = len(list(self.dataset_root_path.glob("train/*/*.avi")))
        video_count_val = len(list(self.dataset_root_path.glob("val/*/*.avi")))
        video_count_test = len(list(self.dataset_root_path.glob("test/*/*.avi")))
        video_total = video_count_train + video_count_val + video_count_test
        self.clip_duration = clip_duration
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,
            pin_memory=True,
        )


class kinetics400DataModule(pytorch_lightning.LightningDataModule):
    # Training dataset transformations.
    def __init__(self,clip_duration):
        self.dataset_root_path = '/home/hong/workspace/datasets/kinetics400_5per/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        self.clip_duration = clip_duration
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            decode_audio=False,
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=_BATCH_SIZE,
            num_workers=_NUM_WORKERS,

        )


class kinetics600DataModule(pytorch_lightning.LightningDataModule):
    # Training dataset transformations.
    def __init__(self,clip_duration):
        self.dataset_root_path = '/home/hong/workspace/datasets/kinetics600_5per/data/'
        self.dataset_root_path = pathlib.Path(self.dataset_root_path)
        video_count_train = len(list(self.dataset_root_path.glob("train/*/*.mp4")))
        video_count_val = len(list(self.dataset_root_path.glob("val/*/*.mp4")))
        video_total = video_count_train + video_count_val
        self.clip_duration = clip_duration
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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
    def __init__(self,clip_duration):
        self.dataset_root_path = '/home/hong/workspace/datasets/hmdb51/data/'
        self.video_path_prefix = self.dataset_root_path
        self.split_id = 1
        self.clip_duration = clip_duration
        # self.dataset_root_path = '/home/hong/workspace/datasets/UCF-101/ii/'
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
        self.dataset_root_path = '/home/hong/workspace/datasets/hmdb51/testTrainMulti_7030_splits'

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
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
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


class SSV2DataModule(pytorch_lightning.LightningDataModule):

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
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(self.resize_to),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),
            ]
        )
        train_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
            video_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/train.json",
            video_path_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/train.csv",
            video_path_prefix="/home/hong/workspace/datasets/ssv2/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
            transform=train_transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
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
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize((self.resize_to, self.resize_to)),
                        ]
                    ),
                ),
            ]
        )
        val_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
            video_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/validation.json",
            video_path_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/val.csv",
            video_path_prefix="/home/hong/workspace/datasets/ssv2/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def test_dataloader(self):
        val_transform = Compose(
            [
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self.num_frames_to_sample),
                            Lambda(lambda x: x / 255.0),
                            Normalize(self.mean, self.std),
                            Resize((self.resize_to, self.resize_to)),
                        ]
                    ),
                ),
            ]
        )
        test_dataset = pytorchvideo.data.SSv2(
            label_name_file="/home/hong/workspace/source/video_mae/datasets/ssv2/labels.json",
            video_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/validation.json",
            video_path_label_file="/home/hong/workspace/source/video_mae/datasets/ssv2/val.csv",
            video_path_prefix="/home/hong/workspace/datasets/ssv2/",
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
            transform=val_transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )
