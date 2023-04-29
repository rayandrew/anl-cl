import json
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from avalanche.benchmarks.generators import nc_benchmark
from avalanche.evaluation.metrics import (  # disk_usage_metrics,
    cpu_usage_metrics,
    gpu_usage_metrics,
    loss_metrics,
    ram_usage_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger
from avalanche.training.checkpoint import (
    maybe_load_checkpoint,
    save_checkpoint,
)
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import (
    AGEM,
    EWC,
    GDumb,
    GSS_greedy,
    LwF,
    Naive,
)

import gorilla
import hydra
from omegaconf import DictConfig, OmegaConf

import src.patches as patches
from src.dataset import (
    AlibabaMachineDataset,
    AlibabaMachineSequenceDataset,
    GoogleMachineDataset,
    GoogleMachineSequenceDataset
)
from src.metrics import (  # forward_transfer_metrics_with_tolerance,
    accuracy_metrics_with_tolerance,
    bwt_metrics_with_tolerance,
    class_accuracy_metrics_with_tolerance,
    class_diff_metrics,
    forgetting_metrics_with_tolerance,
)
from src.models import LSTM, MLP
from src.utils.general import (
    get_checkpoint_fname,
    get_model_fname,
    set_seed,
)
from src.utils.summary import (  # generate_task_table,
    generate_summary,
    generate_summary_table,
)

# @ex.config
# def my_config():
#     # initialize Hydra
#     with initialize(config_path="config", version_base=None):
#         # load the Hydra configuration
#         cfg = compose(config_name="config")

#         # set the configuration for the experiment
#     # ex.config.update(cfg)
#     # unpack all keys of config as separate variables
#     variables = dict(**cfg)
#     locals().update(variables)


def print_and_log(msg, out_file):
    print(msg)
    out_file.write(f"{msg}\n")


@hydra.main(
    config_path="config", config_name="config", version_base=None
)
def main(cfg: DictConfig):
    set_seed(cfg.seed)

    # Patches
    for patch in gorilla.find_patches([patches]):
        gorilla.apply(patch)

    # Output folder
    output_folder = Path(cfg.out_path)
    output_folder.mkdir(exist_ok=True, parents=True)

    # Config
    device = torch.device(
        f"cuda:{cfg.cuda}"
        if torch.cuda.is_available() and cfg.cuda >= 0
        else "cpu"
    )

    # TODO: instantiate this using hydra
    # ChoosenDataset = AlibabaMachineDatasetF
    ChoosenDataset = GoogleMachineSequenceDataset
    train_dataset, raw_train_dataset = ChoosenDataset(
        filename=cfg.filename,
        n_labels=cfg.n_labels,
        subset="training",
        y=cfg.y,
        seq_len=cfg.seq_len,
        univariate=cfg.univariate,
    )

    test_dataset, _raw_test_dataset = ChoosenDataset(
        filename=cfg.filename,
        n_labels=cfg.n_labels,
        subset="testing",
        y=cfg.y,
        seq_len=cfg.seq_len,
        univariate=cfg.univariate,
    )

    # Loggers
    loggers = [
        InteractiveLogger(),
        TextLogger(open(output_folder / "train_log.txt", "w")),
    ]

    # TODO: change this scenario into online setting
    benchmark = nc_benchmark(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        n_experiences=raw_train_dataset.n_experiences(),
        shuffle=False,
        task_labels=False,
    )
    train_stream = benchmark.train_stream
    test_stream = benchmark.test_stream

    # model
    model = (
        LSTM(
            input_size=raw_train_dataset.input_size(),
            num_classes=cfg.n_labels,
            rnn_layers=3,
            hidden_size=512,
            batch_first=True,
        )
        if cfg.sequential
        else MLP(
            input_size=raw_train_dataset.input_size(),
            num_classes=cfg.n_labels,
            hidden_layers=4,
            drop_rate=0.3,
        )
    )

    # Prepare for training & testing
    # optimizer = SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    optimizer = Adam(model.parameters(), lr=cfg.lr)
    criterion = CrossEntropyLoss()
    eval_plugin = EvaluationPlugin(
        loss_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        accuracy_metrics_with_tolerance(
            tolerance=1,
            minibatch=True,
            epoch=True,
            epoch_running=False,
            experience=True,
            stream=True,
        ),
        forgetting_metrics_with_tolerance(
            tolerance=cfg.tolerance, experience=True, stream=True
        ),
        bwt_metrics_with_tolerance(
            tolerance=cfg.tolerance, experience=True, stream=True
        ),
        class_accuracy_metrics_with_tolerance(
            tolerance=cfg.tolerance, experience=True, stream=True
        ),
        # class_diff_metrics(stream=True),
        # forward_transfer_metrics_with_tolerance(
        #     tolerance=1, experience=True, stream=True
        # ),
        # timing_metrics(
        #     minibatch=True,
        #     epoch=False,
        #     epoch_running=False,
        #     experience=True,
        #     stream=True,
        # ),
        # cpu_usage_metrics(
        #     minibatch=True,
        #     epoch=False,
        #     epoch_running=False,
        #     experience=True,
        #     stream=True,
        # ),
        # gpu_usage_metrics(
        #     gpu_id=0,
        #     every=0.5,
        #     minibatch=True,
        #     epoch=True,
        #     experience=True,
        #     stream=True,
        # ),
        # ram_usage_metrics(
        #     every=1,
        #     minibatch=False,
        #     epoch=True,
        #     experience=True,
        #     stream=True,
        # ),
        loggers=loggers,
    )
    print(cfg.strategy)

    if cfg.strategy == "naive":
        cl_strategy = Naive(
            model,
            optimizer,
            criterion,
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            eval_every=0,  # at the end of each experience
        )
    elif cfg.strategy == "agem":
        cl_strategy = AGEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=n_experiences,
            sample_size=32,
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            # eval_every=0,  # at the end of each experience
        )
    elif cfg.strategy == "ewc":
        cl_strategy = EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=0.9,
            mode="separate",
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            # eval_every=0,  # at the end of each experience
        )
    elif cfg.strategy == "lwf":
        cl_strategy = LwF(
            model,
            optimizer,
            criterion,
            alpha=0.5,
            temperature=2,
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            # eval_every=0,  # at the end of each experience
        )
    elif cfg.strategy == "gss":
        cl_strategy = GSS_greedy(
            model,
            optimizer,
            criterion,
            input_size=[raw_train_dataset.input_size()],
            mem_strength=30,
            mem_size=5000,
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            # eval_every=0,  # at the end of each experience
        )
    elif cfg.strategy == "gdumb":
        cl_strategy = GDumb(
            model,
            optimizer,
            criterion,
            mem_size=5000,
            train_mb_size=cfg.batch_size,
            train_epochs=cfg.epoch,
            eval_mb_size=cfg.batch_size,
            evaluator=eval_plugin,
            device=device,
            # eval_every=0,  # at the end of each experience
        )
    else:
        raise ValueError("Strategy not supported")

    checkpoint_fname = output_folder / get_checkpoint_fname(cfg)
    cl_strategy, initial_exp = maybe_load_checkpoint(
        cl_strategy, checkpoint_fname
    )

    # train and test loop
    results = {}
    for exp in train_stream[initial_exp:]:
        cl_strategy.train(
            exp,
            num_workers=cfg.n_workers,
            eval_streams=[test_stream]
        )
        result = cl_strategy.eval(test_stream)
        results[exp.current_experience] = result
        save_checkpoint(cl_strategy, checkpoint_fname)

    json.dump(
        results, open(output_folder / "train_results.json", "w")
    )

    # save the model
    model_name = output_folder / get_model_fname(cfg)
    model_scripted = torch.jit.script(model)
    model_scripted.save(model_name)

    # print results
    out_file = open(output_folder / "train_results.txt", "w")
    for key in results:
        print_and_log(f"Experience #{key}", out_file)
        sorted_results = sorted(
            results[key].keys(), key=lambda x: x.lower().strip()
        )
        for k in sorted_results:
            print_and_log(f"{k.strip()}: {results[key][k]}", out_file)
        print_and_log("", out_file)

    summary = generate_summary(output_folder / "train_results.json")
    table = generate_summary_table(summary)
    print(table.draw())


if __name__ == "__main__":
    # out_file = open(output_folder / "model.pt", "w")
    # print_and_log("", out_file)
    main()
    #     parser = ArgumentParser()
    # parser.add_argument(
    #     "--cuda",
    #     type=int,
    #     default=0,
    #     help="Select zero-indexed cuda device. -1 to use CPU.",
    # )
    #     parser.add_argument("-f", "--filename", type=str, required=True)
    #     parser.add_argument(
    #         "-o", "--output_folder", type=str, default="out"
    #     )
    #     parser.add_argument("-m", "--model_name", type=str, required=True)
    #     parser.add_argument("-nl", "--n_labels", type=int, default=10)
    #     parser.add_argument("-w", "--n_workers", type=int, default=4)
    #     parser.add_argument(
    #         "-y",
    #         type=str,
    #         choices=["cpu", "mem", "disk"],
    #         default="cpu",
    #     )
    #     parser.add_argument("-e", "--epoch", type=int, default=8)
    #     # parser.add_argument("-e", "--epoch", type=int, default=1)
    #     parser.add_argument(
    #         "-lr", "--learning_rate", type=float, default=0.001
    #     )
    #     parser.add_argument("-b", "--batch_size", type=int, default=32)
    #     parser.add_argument(
    #         "-s",
    #         "--strategy",
    #         type=str,
    #         choices=["gss", "agem", "naive", "lwf", "ewc", "gdumb"],
    #         default="gdumb",
    #     )
    #     parser.add_argument("--seq", action="store_true")
    #     parser.add_argument("--seq_len", type=int, default=3)
    #     parser.add_argument("--univariate", action="store_true")
    #     parser.add_argument("--local", action="store_true")
    #     args = parser.parse_args()
    #     main(args)
