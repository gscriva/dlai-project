import argparse


def main_parser():
    # Parse arguments and prepare program
    parser = argparse.ArgumentParser(description="Training Custom NN")
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to .pth file checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="use this flag to validate without training",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        metavar="N",
        help="batch size (default: 100)",
    )
    parser.add_argument(
        "--test_batch_size",
        default=200,
        type=int,
        metavar="N",
        help="batch size (default: 200)",
    )
    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        metavar="N",
        help="number of epochs (default: 100)",
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
        default=0,
        type=float,
        metavar="N",
        help="learning rate (default 0)",
    )
    parser.add_argument(
        "--data_dir",
        default="dataset/train_data_L14.npz",
        type=str,
        metavar="string",
        help="dataset directory (used for both train and validation)",
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
        default=None,
        type=int,
        metavar="N",
        help="size of input (if smaller than its natural size)",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        metavar="N",
        help="number of CPU to use in training",
    )
    parser.add_argument(
        "--model_type",
        default="CNN",
        type=str,
        metavar="string",
        help="type of the model to train/eval",
    )
    return parser
