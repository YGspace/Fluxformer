import load_model_mvit2_flow as mvit
#import load_model_vivit
import dataloader_all as dataloader
import pytorch_lightning as pl

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import torch
from utils.parser import load_config, parse_args
from config.defaults import assert_and_infer_cfg

def train_model():

    args = parse_args()
    print("config files: {}".format(args.cfg_files))
    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)

    train_loader, val_loader, num_classes,max_iter = dataloader.make_dataloader(args.data_name)
    model = mvit.VideoClassificationLightningModule(num_classes, lr=3e-4,args=args,cfg=cfg,max_iter=max_iter)


    print(model)
    trainer = pl.Trainer(
        accelerator="gpu",
        gpus=1 if str(device) == "cuda:0" else 0,
        check_val_every_n_epoch=5,
        max_epochs=180,
        auto_lr_find=True,
        callbacks=[
            ModelCheckpoint(dirpath="checkpoints/",save_weights_only=True, save_last=True,mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    trainer.fit(model, train_loader, val_loader)

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    #test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}

    return model, result

if __name__ == '__main__':
#"UCF101",  "kinetics400", "kinetics600", "HMDB51":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    train_model()


