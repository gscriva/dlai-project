import argparse


def parser():
    # Parse arguments and prepare program
    parser = argparse.ArgumentParser(description="Training Custom NN")

    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to .pth file checkpoint",
    )
    parser.add_argument(
        "-t",
        "--train",
        dest="train",
        action="store_true",
        help="use this flag to train the model",
    )
    parser.add_argument(
        "-d",
        "--dropout",
        dest="dropout",
        action="store_true",
        help="use this flag to use dropout in the model",
    )
    parser.add_argument(
        "-b",
        "--batchnorm",
        dest="batchnorm",
        action="store_true",
        help="use this flag to use batchnorm in the model",
    )
    parser.add_argument(
        "-w",
        "--wandb",
        dest="save_wandb",
        action="store_true",
        help="use this flag to save the model on wandb",
    )
    parser.add_argument(
        "-s",
        "--scheduler",
        dest="scheduler",
        action="store_true",
        help="use this flag to use scheduler during training",
    )
    parser.add_argument(
        "--norm",
        dest="normalize",
        action="store_true",
        help="use this flag to normalize the dataset",
    )
    parser.add_argument(
        "--stand",
        dest="standardize",
        action="store_true",
        help="use this flag to standardize the dataset",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        metavar="N",
        help="training batch size (default: 100)",
    )
    parser.add_argument(
        "--val_batch_size",
        default=2000,
        type=int,
        metavar="N",
        help="validation batch size [Set 0 for testing] (default: 2000)",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of epochs (default: 2)",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        metavar="N",
        help="learning rate (default 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        metavar="N",
        help="learning rate (default 0.)",
    )
    parser.add_argument(
        "--activation",
        default="rrelu",
        type=str,
        metavar="string",
        help="specify activation function (default rrelu)",
    )
    parser.add_argument(
        "--data_dir",
        nargs="+",
        type=str,
        help=" list of datasets directory (used for both train and validation)",
    )
    parser.add_argument(
        "--weights_path",
        default=None,
        type=str,
        metavar="PATH",
        help="dataset directory used to load model's parameters (default: None)",
    )
    parser.add_argument(
        "--train_size",
        default=0.9,
        type=float,
        help="Size (in percent) of the training set (default: 0.9)",
    )
    parser.add_argument(
        "--input_name",
        default="speckleF",
        type=str,
        metavar="string",
        help="specify input name (default speckleF)",
    )
    parser.add_argument(
        "--output_name",
        default="evalues",
        type=str,
        metavar="string",
        help="specify target name in dataset (default evalues)",
    )
    parser.add_argument(
        "--input_size",
        nargs="+",
        type=int,
        help="list of input sizes (if smaller than its natural size)",
        required=True,
    )
    parser.add_argument(
        "--nofreeze_layer",
        nargs="+",
        type=str,
        help="list of layer to not freeze",
        default=None,
    )
    parser.add_argument(
        "--hidden_dim",
        default=128,
        type=int,
        metavar="N",
        help="channels in the hidden layers (default 128)",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        metavar="N",
        help="number of the hidden layers (default 3)",
    )
    parser.add_argument(
        "--kernel_size",
        default=None,
        type=int,
        metavar="N",
        help="size of the kernel (default None)",
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of CPU to use in training (default 8)",
    )
    parser.add_argument(
        "--model_type",
        default="MLP",
        type=str,
        metavar="string",
        help="type of the model to train/eval (default MLP)",
    )
    return parser
