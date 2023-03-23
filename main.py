from argparse import ArgumentParser
from pathlib import Path
import json

import gorilla

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from avalanche.benchmarks.generators import nc_benchmark

from avalanche.models import SimpleMLP
from avalanche.training.plugins import EvaluationPlugin

from avalanche.training.supervised import Naive, AGEM, LwF, EWC, GSS_greedy, GDumb
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger

# from avalanche.benchmarks.utils import AvalancheDataset

# from model import SimpleMLP
from dataset import AlibabaSchedulerDataset, AlibabaMachineDataset 
from utils import process_file, generate_table

import patches


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

    raw_train_dataset = AlibabaSchedulerDataset(
        filename=args.filename, n_labels=args.n_labels, mode="train", y=args.y
    )
    raw_test_dataset = AlibabaSchedulerDataset(
        filename=args.filename, n_labels=args.n_labels, mode="test", y=args.y
    )

    train_dataset = raw_train_dataset
    test_dataset = raw_test_dataset

    # train_dataset = AvalancheDataset([raw_train_dataset])
    # train_dataset.targets = raw_train_dataset.targets

    # test_dataset = AvalancheDataset([raw_test_dataset])
    # test_dataset.targets = raw_test_dataset.targets

    # model
    model = SimpleMLP(
        # input_size=len(AlibabaDataset.FEATURE_COLUMNS),
        input_size=raw_train_dataset.get_input_size(),
        # input_size=47,
        num_classes=args.n_labels,
        hidden_layers=4,
        drop_rate=0.3,
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
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = CrossEntropyLoss()
    eval_plugin = EvaluationPlugin(
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        accuracy_metrics(
            minibatch=True, epoch=True, epoch_running=True, experience=True, stream=True
        ),
        # forgetting_metrics(experience=True, stream=True),
        loggers=loggers,
    )

    if args.strategy == "naive":
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
    elif args.strategy == "agem":
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
    elif args.strategy == "ewc":
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=0.3,
            mode="separate",
            train_mb_size=args.batch_size,
            train_epochs=args.epoch,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    elif args.strategy == "lwf":
        cl_strategy = LwF(
            model,
            optimizer,
            criterion,
            alpha=0.5,
            temperature=2,
            train_mb_size=args.batch_size,
            train_epochs=args.epoch,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    elif args.strategy == "gss":
        cl_strategy = GSS_greedy(
            model,
            optimizer,
            criterion,
            # input_size=[len(AlibabaDataset.FEATURE_COLUMNS)],
            input_size=[raw_train_dataset.get_input_size()],
            # input_size=[47],
            mem_strength=30,
            mem_size=5000,
            train_mb_size=args.batch_size,
            train_epochs=args.epoch,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    elif args.strategy == "gdumb":
        cl_strategy = GDumb(
            model,
            optimizer,
            criterion,
            mem_size=5000,
            train_mb_size=args.batch_size,
            train_epochs=args.epoch,
            eval_mb_size=args.batch_size,
            evaluator=eval_plugin,
            device=device,
        )
    else:
        raise ValueError("Strategy not supported")

    # train and test loop
    results = {}
    for exp in train_stream:
        cl_strategy.train(exp, num_workers=args.n_workers)
        result = cl_strategy.eval(test_stream)
        results[exp.current_experience] = result

    json.dump(results, open(output_folder / "train_results.json", "w"))

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

    summary = {}
    summary[args.strategy] = process_file(output_folder / "train_results.json")
    table = generate_table(data=summary)
    print(table.draw())


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
    parser.add_argument("-e", "--epoch", type=int, default=8)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument(
        "-s",
        "--strategy",
        type=str,
        choices=["gss", "agem", "naive", "lwf", "ewc", "gdumb"],
        default="gdumb",
    )
    args = parser.parse_args()
    main(args)
