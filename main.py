from argparse import ArgumentParser
from pathlib import Path

import gorilla

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

from avalanche.benchmarks.generators import nc_benchmark
from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, AGEM
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger

# from avalanche.benchmarks.utils import AvalancheDataset

from dataset import AlibabaDataset

import patches


def prettify_result(d: dict) -> str:
    return ""


def print_and_log(msg, out_file):
    print(msg)
    out_file.write(f"{msg}\n")


def main(args):
    # Patches
    for patch in gorilla.find_patches([patches]):
        gorilla.apply(patch)

    # Output folder
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

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
    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_folder / "train_log.txt", "w")),
    ]

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
    # cl_strategy = Naive(
    #     model,
    #     optimizer,
    #     criterion,
    #     train_mb_size=args.batch_size,
    #     train_epochs=args.epoch,
    #     eval_mb_size=args.batch_size,
    #     evaluator=eval_plugin,
    #     device=device,
    # )

    cl_strategy = AGEM(
        model,
        optimizer,
        criterion,
        patterns_per_exp=args.n_experiences,
        sample_size=32,
        train_mb_size=args.batch_size,
        train_epochs=args.epoch,
        eval_mb_size=args.batch_size,
        evaluator=eval_plugin,
        device=device,
    )

    # train and test loop
    results = {}
    for exp in train_stream:
        cl_strategy.train(exp, num_workers=args.n_workers)
        result = cl_strategy.eval(test_stream)
        results[exp.current_experience] = result

    # Saving model
    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save(output_folder / f"{args.model_name}.pt")  # Save

    # print results
    out_file = open(output_folder / "train_results.txt", "w")
    for key in results:
        print_and_log(f"Experience #{key}", out_file)
        sorted_results = sorted(results[key].keys(), key=lambda x: x.lower().strip())
        for k in sorted_results:
            print_and_log(f"{k.strip()}: {results[key][k]}", out_file)
        print_and_log("", out_file)

    print(prettify_result(results))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--filename", type=str, required=True)
    parser.add_argument("-o", "--output_folder", type=str, default="out")
    parser.add_argument("-m", "--model_name", type=str, required=True)
    parser.add_argument("-x", "--n_experiences", type=int, required=True)
    parser.add_argument("-nl", "--n_labels", type=int, default=10)
    parser.add_argument("-w", "--n_workers", type=int, default=4)
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
