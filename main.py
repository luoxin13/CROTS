import os
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as callbacks

from modules import find_module_using_name
from argparse import ArgumentParser, Namespace
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    params = ArgumentParser()
    params.add_argument("--module-name", type=str, default="basic")
    params.add_argument("--mode", type=str, default="test")
    params.add_argument("--val-interval", type=float, default=1.0)
    params.add_argument("--log-interval", type=int, default=20)
    tmp_params, _ = params.parse_known_args()
    #
    module_cls, data_module_cls = find_module_using_name(tmp_params.module_name)
    params = module_cls.add_params(params)
    params = data_module_cls.add_params(params)
    #
    params = params.parse_args()
    try:
        restore_from = getattr(params, "restore_from")
        if not os.path.isfile(restore_from) and not restore_from.startswith("http"):
            params.restore_from = None
    except AttributeError:
        setattr(params, restore_from, None)
    module = module_cls(params)
    data_module = data_module_cls(params)
    
    os.makedirs(os.path.join(params.log_tb_dir, params.module_name, params.experiment_name), exist_ok=True)
    logger = TensorBoardLogger(
        save_dir=os.path.join(params.log_tb_dir, params.module_name),
        name=params.experiment_name,
    )
    ckpt_callback = callbacks.ModelCheckpoint(
        filename='checkpoint-step_{step}-miou_{val_miou:.2f}',
        auto_insert_metric_name=False,
        monitor="val_miou",
        mode="max",
    )
    #
    trainer = pl.Trainer(
        gpus=torch.cuda.device_count(),
        strategy='ddp',
        benchmark=True,
        logger=logger,
        log_every_n_steps=params.log_interval,
        sync_batchnorm=True,
        val_check_interval=params.val_interval,
        callbacks=[
            ckpt_callback,
        ],
    )
    
    if params.mode == "test":
        trainer.test(model=module, datamodule=data_module)
    else:
        trainer.fit(model=module, datamodule=data_module)


if __name__ == "__main__":
    main()
