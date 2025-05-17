import os
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import torch
import faulthandler
import numpy as np
import time
from collections import OrderedDict
from torch.cuda.amp import autocast as autocast

faulthandler.enable()
import utils
from modules.sync_batchnorm import convert_model
from seq_scripts import seq_train, seq_eval, seq_feature_generation
from utils.parameters import ConfigArgs
from utils.optimizer import Optimizer
from dataset.dataloader_video import BaseFeeder
from slr_network import SLRModel

class Processor:
    def __init__(self, arg: ConfigArgs):
        self.arg = arg
        self.setup_environment()
        self.load_gloss_dict()
        self.model, self.optimizer = self.load_model()
        self.load_data()

    def setup_environment(self):
        self.device = utils.GpuDataParallel()
        self.recoder = utils.Recorder(self.arg.work_dir, self.arg.print_log, self.arg.log_interval)

        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)

        self.save_arg()
        if self.arg.random_fix:
            self.rng = utils.RandomState(seed=self.arg.random_seed)

    def save_arg(self):
        with open(os.path.join(self.arg.work_dir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(self.arg), f)

    def load_gloss_dict(self):
        self.gloss_dict = np.load(self.arg.dataset_info['dict_path'], allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1

    def load_model(self):
        self.device.set_device(self.arg.device)
        model = SLRModel(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        optimizer = Optimizer(model, self.arg.optimizer_args)

        # Load weights if specified
        if self.arg.load_weights:
            self._load_weights(model, self.arg.load_weights)
        elif self.arg.load_checkpoints:
            self._load_checkpoint(model, optimizer)

        model = self._to_device(model)
        self.kernel_sizes = model.conv1d.kernel_size
        return model, optimizer

    def _load_weights(self, model: SLRModel, path: str):
        weights = torch.load(path, weights_only=False)['model_state_dict']
        clean_weights = OrderedDict((k.replace('.module', ''), v) for k, v in weights.items())
        model.load_state_dict(clean_weights, strict=True)

    def _load_checkpoint(self, model: SLRModel, optimizer: Optimizer):
        checkpoint = torch.load(self.arg.load_checkpoints)
        self._load_weights(model, self.arg.load_checkpoints)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.arg.optimizer_args['start_epoch'] = checkpoint['epoch'] + 1

    def _to_device(self, model: SLRModel):
        model = model.to(self.device.output_device)
        model = convert_model(model)
        return model.cuda()

    def load_data(self):
        self.dataset = {}
        self.data_loader = {}

        dataset_modes = {
            "phoenix": [("train", True), ("train_eval", False), ("dev", False), ("test", False)],
            "how2sign": [("train", True), ("dev", False), ("test", False)],
        }

        for mode, is_train in dataset_modes.get(self.arg.dataset, []):
            args = self.arg.feeder_args
            args.update({
                "prefix": self.arg.dataset_info['dataset_root'],
                "mode": mode.split("_")[0],
                "transform_mode": is_train,
            })
            dataset = BaseFeeder(gloss_dict=self.gloss_dict, kernel_size=self.kernel_sizes,
                                   dataset=self.arg.dataset, **args)
            self.dataset[mode] = dataset
            self.data_loader[mode] = self._build_dataloader(dataset, mode, is_train)

    def _build_dataloader(self, dataset: BaseFeeder, mode: str, train_flag: bool):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if train_flag else self.arg.test_batch_size,
            shuffle=train_flag,
            drop_last=train_flag,
            num_workers=self.arg.num_worker,
            prefetch_factor=1,
            persistent_workers=True,
            collate_fn=self.dataset[mode].collate_fn,
            pin_memory=True,
            worker_init_fn=lambda wid: np.random.seed(self.arg.random_seed + wid)
        )

    def start(self):
        if self.arg.phase in ['train', 'dev']:
            self.train()
        elif self.arg.phase == 'test':
            self.evaluate()
        elif self.arg.phase == 'features':
            self.extract_features()

    def train(self):
        best_wer = 100.0
        for epoch in range(self.arg.optimizer_args['start_epoch'], self.arg.num_epoch):
            start_time = time.time()

            seq_train(self.data_loader[self.arg.phase], self.model, self.optimizer,
                      self.device, epoch, self.recoder)

            if epoch % self.arg.eval_interval == 0:
                wer = seq_eval(self.arg, self.data_loader['dev'], self.model,
                               self.device, 'dev', epoch, self.arg.work_dir, self.recoder,
                               self.arg.evaluate_tool)

                if wer < best_wer:
                    best_wer = wer
                    self.save_model(epoch, f"{self.arg.work_dir}_best_model.pt")

            if epoch % self.arg.save_interval == 0:
                self.save_model(epoch, f"{self.arg.work_dir}/epoch{epoch}_model.pt")

            duration = time.time() - start_time
            self.recoder.log(f"Epoch {epoch} took {int(duration)//60}m {int(duration)%60}s")

    def evaluate(self):
        self.recoder.log("Evaluating model")
        for mode in ['dev', 'test']:
            seq_eval(self.arg, self.data_loader[mode], self.model,
                     self.device, mode, 6667, self.arg.work_dir, self.recoder,
                     self.arg.evaluate_tool)

    def extract_features(self):
        for mode in ['train', 'dev', 'test']:
            loader = self.data_loader[mode if mode != 'train' else 'train_eval']
            seq_feature_generation(loader, self.model, self.device, mode, self.arg.work_dir, self.recoder)

    def save_model(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.optimizer.scheduler.state_dict(),
            'rng_state': self.rng.save_rng_state() if hasattr(self, 'rng') else None,
        }, path)

if __name__ == '__main__':
    # Parse arguments
    parser = utils.get_parser()
    args = parser.parse_args()

    # Load YAML config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_args = yaml.load(f, Loader=yaml.FullLoader)

        # Merge config file values into parser defaults
        for key, value in config_args.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise ValueError(f"Invalid config key: {key}")

    # Load dataset-specific info
    dataset_config_path = os.path.join("configs", f"{args.dataset}.yaml")
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"Dataset config file not found: {dataset_config_path}")
    with open(dataset_config_path, 'r') as f:
        args.dataset_info = yaml.load(f, Loader=yaml.FullLoader)

    # Initialize and run the processor
    processor = Processor(args)
    processor.start()

