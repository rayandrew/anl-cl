from argparse import ArgumentParser

import gorilla

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.generators import nc_benchmark

# from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger

from dataset import AlibabaDataset

import patches


def main(args):
    # Patches
    for patch in gorilla.find_patches([patches]):
        gorilla.apply(patch)

    # Config
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw_train_dataset = AlibabaDataset(
        filename=args.filename, n_labels=args.n_labels, train=True, y=args.y
    )
    raw_test_dataset = AlibabaDataset(
        filename=args.filename, n_labels=args.n_labels, train=False, y=args.y
    )

    train_dataset = raw_train_dataset
    test_dataset = raw_test_dataset

    # train_dataset = AvalancheDataset([raw_train_dataset])
    # train_dataset.targets = raw_train_dataset.targets

    # test_dataset = AvalancheDataset([raw_test_dataset])
    # test_dataset.targets = raw_test_dataset.targets

    # model
    model = SimpleMLP(
        input_size=len(AlibabaDataset.FEATURE_COLUMNS), num_classes=args.n_labels
    )

    # Loggers
    loggers = [InteractiveLogger()]

    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=args.n_experiences,
        shuffle=False,
        task_labels=False,
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # Prepare for training & testing
    optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = CrossEntropyLoss()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(
            minibatch=True, epoch=True, epoch_running=True, experience=True, stream=True
        ),
        forgetting_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    # Continual learning strategy
    cl_strategy = Naive(
        model,
        optimizer,
        criterion,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        evaluator=eval_plugin,
        device=device,
    )

    # train and test loop
    results = {}
    for exp in train_stream:
        cl_strategy.train(exp, num_workers=4)
        result = cl_strategy.eval(test_stream)
        results[exp.current_experience] = result

    # print results
    for key in results:
        print(f"Experience #{key}")
        sorted_results = sorted(results[key].keys(), key=lambda x: x.lower().strip())
        for k in sorted_results:
            print(f"{k.strip()}: {results[key][k]}")
        print()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-x", "--n_experiences", type=int, required=True)
    parser.add_argument("-nl", "--n_labels", type=int, default=10)
    parser.add_argument(
        "-y",
        type=str,
        choices=["cpu_util_percent", "mem_util_percent", "disk_io_percent"],
        default="cpu_util_percent",
    )
    parser.add_argument("-e", "--epoch", type=int, default=4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
