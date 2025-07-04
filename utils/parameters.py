import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

@dataclass
class ConfigArgs:
    # general settings
    work_dir: str = './work_dir/temp'
    config: str = './configs/baseline.yaml'
    random_fix: bool = True
    # device is defined as str in the parser, so we use str here (even though its default is 0)
    device: str = "0"
    
    # processor
    phase: str = 'train'
    
    # debug settings
    save_interval: int = 200
    random_seed: int = 0
    eval_interval: int = 100
    print_log: bool = True
    log_interval: int = 20
    evaluate_tool: str = "python"
    
    # feeder configuration
    feeder: str = 'dataloader_video.BaseFeeder'
    dataset: Optional[str] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    num_worker: int = 4
    feeder_args: Dict[str, Any] = field(default_factory=dict)
    
    # model settings
    model: Optional[str] = None
    model_args: Dict[str, Any] = field(default_factory=dict)
    load_weights: Optional[str] = None
    load_checkpoints: Optional[str] = None
    decode_mode: str = "max"
    ignore_weights: List[str] = field(default_factory=list)
    
    # optimization and training
    batch_size: int = 16
    test_batch_size: int = 8
    loss_weights: Dict[str, float] = field(default_factory=lambda: {"SeqCTC": 1.0})
    optimizer_args: Dict[str, Any] = field(default_factory=lambda: {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [5, 10],
        "weight_decay": 0.00005,
        "start_epoch": 1,
    })
    num_epoch: int = 80


def get_parser():
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='The pytorch implementation for Visual Alignment Constraint '
                    'for Continuous Sign Language Recognition.')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')
    parser.add_argument(
        '--config',
        default='./configs/baseline.yaml',
        help='path to the configuration file')
    parser.add_argument(
        '--random_fix',
        type=str2bool,
        default=True,
        help='fix random seed or not')
    parser.add_argument(
        '--device',
        type=str,
        default=0,
        help='the indexes of GPUs for training or testing')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # processor
    parser.add_argument(
        '--phase', default='train', help='can be train, test and features')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # debug
    parser.add_argument(
        '--save-interval',
        type=int,
        default=200,
        help='the interval for storing models (#epochs)')
    parser.add_argument(
        '--random-seed',
        type=int,
        default=0,
        help='the default value for random seed.')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epochs)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=20,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--evaluate-tool', default="python", help='sclite or python')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # feeder
    parser.add_argument(
        '--feeder', default='dataloader_video.BaseFeeder', help='data loader will be used')
    parser.add_argument(
        '--dataset',
        default=None,
        help='data loader will be used'
    )
    parser.add_argument(
        '--dataset-info',
        default=dict(),
        help='data loader will be used'
    )
    parser.add_argument(
        '--num-worker',
        type=int,
        default=4,
        help='the number of worker for data loader')
    parser.add_argument(
        '--feeder-args',
        default=dict(),
        help='the arguments of data loader')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--load-weights',
        default=None,
        help='load weights for network initialization')
    parser.add_argument(
        '--load-checkpoints',
        default=None,
        help='load checkpoints for continue training')
    parser.add_argument(
        '--decode-mode',
        default="max",
        help='search mode for decode, max or beam')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # optim
    parser.add_argument(
        '--batch-size', type=int, default=16, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=8, help='test batch size')

    default_optimizer_dict = {
        "base_lr": 1e-2,
        "optimizer": "SGD",
        "nesterov": False,
        "step": [5, 10],
        "weight_decay": 0.00005,
        "start_epoch": 1,
    }
    default_loss_dict = {
        "SeqCTC": 1.0,
    }

    parser.add_argument(
        '--loss-weights',
        default=default_loss_dict,
        help='loss selection'
    )

    parser.add_argument(
        '--optimizer-args',
        default=default_optimizer_dict,
        help='the arguments of optimizer')

    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    return parser


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
