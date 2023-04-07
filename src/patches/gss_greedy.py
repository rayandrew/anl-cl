import torch

from avalanche.benchmarks.utils import make_classification_dataset
from avalanche.benchmarks.utils.data_loader import ReplayDataLoader
from avalanche.training.plugins.gss_greedy import GSS_greedyPlugin
from avalanche.training.templates import SupervisedTemplate

import gorilla

from src.utils.patch import patch_filter


@gorilla.patches(
    GSS_greedyPlugin,
    gorilla.Settings(allow_hit=True),
    filter=patch_filter,
)
class CustomGSS_greedyPlugin:
    def __init__(
        self: GSS_greedyPlugin,
        mem_size=200,
        mem_strength=5,
        input_size=[],
    ):
        """

        :param mem_size: total number of patterns to be stored
            in the external memory.
        :param mem_strength:
        :param input_size:
        """
        super(GSS_greedyPlugin, self).__init__()
        self.mem_size = mem_size
        self.mem_strength = mem_strength
        self.device = torch.device("cpu")

        self.ext_mem_list_x = torch.FloatTensor(
            mem_size, *input_size
        ).fill_(0)
        self.ext_mem_list_d = torch.LongTensor(mem_size).fill_(
            0
        )  # added
        self.ext_mem_list_y = torch.LongTensor(mem_size).fill_(0)
        self.ext_mem_list_current_index = 0

        self.buffer_score = torch.FloatTensor(self.mem_size).fill_(0)

    def before_training(
        self: GSS_greedyPlugin,
        strategy: "SupervisedTemplate",
        **kwargs,
    ):
        self.device = strategy.device
        self.ext_mem_list_x = self.ext_mem_list_x.to(strategy.device)
        self.ext_mem_list_d = self.ext_mem_list_d.to(
            strategy.device
        )  # added
        self.ext_mem_list_y = self.ext_mem_list_y.to(strategy.device)
        self.buffer_score = self.buffer_score.to(strategy.device)

    def before_training_exp(
        self: GSS_greedyPlugin,
        strategy,
        num_workers=0,
        shuffle=True,
        **kwargs,
    ):
        """
        Dataloader to build batches containing examples from both memories and
        the training dataset
        """
        if self.ext_mem_list_current_index == 0:
            return

        temp_x_tensors = []
        for elem in self.ext_mem_list_x:
            temp_x_tensors.append(elem.to("cpu"))
        temp_d_tensors = self.ext_mem_list_d.to("cpu")  # added
        temp_y_tensors = self.ext_mem_list_y.to("cpu")

        memory = list(
            zip(temp_x_tensors, temp_d_tensors, temp_y_tensors)
        )  # added
        memory = make_classification_dataset(
            memory, targets=temp_d_tensors
        )

        strategy.dataloader = ReplayDataLoader(
            strategy.adapted_dataset,
            memory,
            oversample_small_tasks=True,
            num_workers=num_workers,
            batch_size=strategy.train_mb_size,
            shuffle=shuffle,
        )

    def after_forward(
        self: GSS_greedyPlugin,
        strategy,
        num_workers=0,
        shuffle=True,
        **kwargs,
    ):
        """
        After every forward this function select sample to fill
        the memory buffer based on cosine similarity
        """

        strategy.model.eval()

        # Compute the gradient dimension
        grad_dims = []
        for param in strategy.model.parameters():
            grad_dims.append(param.data.numel())

        place_left = (
            self.ext_mem_list_x.size(0)
            - self.ext_mem_list_current_index
        )
        if place_left <= 0:  # buffer full
            batch_sim, mem_grads = self.get_batch_sim(
                strategy,
                grad_dims,
                batch_x=strategy.mb_x,
                batch_y=strategy.mb_y,
            )

            if batch_sim < 0:
                buffer_score = self.buffer_score[
                    : self.ext_mem_list_current_index
                ].cpu()

                buffer_sim = (
                    buffer_score - torch.min(buffer_score)
                ) / (
                    (
                        torch.max(buffer_score)
                        - torch.min(buffer_score)
                    )
                    + 0.01
                )

                # draw candidates for replacement from the buffer
                index = torch.multinomial(
                    buffer_sim,
                    strategy.mb_x.size(0),
                    replacement=False,
                ).to(strategy.device)

                # estimate the similarity of each sample in the received batch
                # to the randomly drawn samples from the buffer.
                batch_item_sim = self.get_each_batch_sample_sim(
                    strategy,
                    grad_dims,
                    mem_grads,
                    strategy.mb_x,
                    strategy.mb_y,
                )

                # normalize to [0,1]
                scaled_batch_item_sim = (
                    (batch_item_sim + 1) / 2
                ).unsqueeze(1)
                buffer_repl_batch_sim = (
                    (self.buffer_score[index] + 1) / 2
                ).unsqueeze(1)
                # draw an event to decide on replacement decision
                outcome = torch.multinomial(
                    torch.cat(
                        (
                            scaled_batch_item_sim,
                            buffer_repl_batch_sim,
                        ),
                        dim=1,
                    ),
                    1,
                    replacement=False,
                )
                # replace samples with outcome =1
                added_indx = torch.arange(
                    end=batch_item_sim.size(0), device=strategy.device
                )
                sub_index = outcome.squeeze(1).bool()
                self.ext_mem_list_x[index[sub_index]] = strategy.mb_x[
                    added_indx[sub_index]
                ].clone()
                self.ext_mem_list_d[index[sub_index]] = strategy.mb_d[
                    added_indx[sub_index]
                ].clone()  # added
                self.ext_mem_list_y[index[sub_index]] = strategy.mb_y[
                    added_indx[sub_index]
                ].clone()
                self.buffer_score[index[sub_index]] = batch_item_sim[
                    added_indx[sub_index]
                ].clone()
        else:
            offset = min(place_left, strategy.mb_x.size(0))
            updated_mb_x = strategy.mb_x[:offset]
            updated_mb_d = strategy.mb_d[:offset]  # @rayandrew added
            updated_mb_y = strategy.mb_y[:offset]

            # first buffer insertion
            if self.ext_mem_list_current_index == 0:
                batch_sample_memory_cos = (
                    torch.zeros(updated_mb_x.size(0)) + 0.1
                )
            else:
                # draw random samples from buffer
                mem_grads = self.get_rand_mem_grads(
                    strategy=strategy,
                    grad_dims=grad_dims,
                    gss_batch_size=len(strategy.mb_x),
                )

                # estimate a score for each added sample
                batch_sample_memory_cos = (
                    self.get_each_batch_sample_sim(
                        strategy,
                        grad_dims,
                        mem_grads,
                        updated_mb_x,
                        updated_mb_y,
                    )
                )

            curr_idx = self.ext_mem_list_current_index
            self.ext_mem_list_x[
                curr_idx : curr_idx + offset
            ].data.copy_(updated_mb_x)
            self.ext_mem_list_d[
                curr_idx : curr_idx + offset
            ].data.copy_(
                updated_mb_d
            )  # added
            self.ext_mem_list_y[
                curr_idx : curr_idx + offset
            ].data.copy_(updated_mb_y)
            self.buffer_score[
                curr_idx : curr_idx + offset
            ].data.copy_(batch_sample_memory_cos)
            self.ext_mem_list_current_index += offset

        strategy.model.train()


__all__ = ["CustomGSS_greedyPlugin"]
