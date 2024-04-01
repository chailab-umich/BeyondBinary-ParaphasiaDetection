#!/usr/bin/env/python3
"""
Freeze OG char-model
Train Transformer ASR decoder to predict tokens + paraphasia label using

Recipe for training a wav2vec-based ctc ASR system with librispeech.
The system employs wav2vec as its encoder. Decoding is performed with
ctc greedy decoder.
To run this recipe, do the followering:
> python train_with_wav2vec.py hparams/train_{hf,sb}_wav2vec.yaml
The neural network is trained on CTC likelihood target and character units
are used as basic recognition tokens.

Authors
 * Rudolf A Braun 2022
 * Titouan Parcollet 2022
 * Sung-Lin Yeh 2021
 * Ju-Chieh Chou 2020
 * Mirco Ravanelli 2020
 * Abdel Heba 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
"""

from torch.utils.data import DataLoader
from speechbrain.dataio.dataloader import LoopedLoader
import numpy as np
from scipy.io import wavfile
import wave
from tqdm import tqdm
import librosa
import pandas as pd
import math
import os
import sys
import torch
import logging
import speechbrain as sb

# import speechbrain.speechbrain as sb
from speechbrain.utils.distributed import run_on_main
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from datasets import load_dataset, load_metric, Audio
import re
import time
from speechbrain.tokenizers.SentencePiece import SentencePiece

# multi-gpu
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import seaborn as sns
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)


class DynamicLossScaler:
    def __init__(self, alpha=0.8):
        self.alpha = alpha
        self.avg_loss1 = 1.0
        self.avg_loss2 = 1.0

    def update_avg_losses(self, loss1, loss2):
        self.avg_loss1 = (
            self.alpha * self.avg_loss1 + (1 - self.alpha) * loss1.item()
        )
        self.avg_loss2 = (
            self.alpha * self.avg_loss2 + (1 - self.alpha) * loss2.item()
        )

    def scale_losses(self, loss1, loss2):
        scaled_loss1 = loss1 / self.avg_loss1
        scaled_loss2 = loss2 / self.avg_loss2
        return scaled_loss1, scaled_loss2


def props(cls):
    return [i for i in cls.__dict__.keys() if i[:1] != "_"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_majority_element(lst):
    counts = {}
    for num in lst:
        counts[num] = counts.get(num, 0) + 1
    return max(counts, key=counts.get)


def combine_AWER(hyps_para, hyps_asr, predicted_words, tokenizer, ptokenizer):
    tokenized_out = [tokenizer.IdToPiece(h) for h in hyps_asr]
    assert len(tokenized_out) == len(
        hyps_para
    ), f"Error arrs are of same size:\tokenized_out: {tokenized_out}\hyps_para: {hyps_para}"
    # print(f"tokenized_out: {tokenized_out}")
    p_result = []
    multitoken_result = []
    for t, p in zip(tokenized_out, hyps_para):
        if t == "▁":
            # start of multi-token word
            if len(multitoken_result) > 0:
                # p_result.append(int(sum(multitoken_result) > 1))
                p_result.append(max(multitoken_result))

            multitoken_result = []
        elif t.startswith("▁"):
            if len(multitoken_result) > 0:
                # p_result.append(int(sum(multitoken_result) > 2))
                p_result.append(max(multitoken_result))
            multitoken_result = [p]

        else:
            multitoken_result.append(p)

    # add final multitoken result
    if len(multitoken_result) > 0:
        # p_result.append(int(sum(multitoken_result) > 1))
        p_result.append(max(multitoken_result))

    assert len(predicted_words) == len(
        p_result
    ), f"Error arrs are of same size:\npredicted_words: {predicted_words}\np_result: {p_result}"
    pred_AWER_list = []
    ptokenizer = [c.upper() for c in ptokenizer]
    for w, p in zip(predicted_words, p_result):
        pred_AWER_list.append(f"{w}/{ptokenizer[p]}")

    return [pred_AWER_list]


def combine_F1(hyps_para, hyps_asr, tokenizer):
    tokenized_out = [tokenizer.IdToPiece(h) for h in hyps_asr]
    # print(f"tokenized_out: {tokenized_out}")
    assert len(tokenized_out) == len(
        hyps_para
    ), f"Error arrs are of same size:\tokenized_out: {tokenized_out}\hyps_para: {hyps_para}"

    p_result = []
    multitoken_result = []
    for t, p in zip(tokenized_out, hyps_para):
        if t == "▁":
            # start of multi-token word
            if len(multitoken_result) > 0:
                # p_result.append(find_majority_element(multitoken_result))
                p_result.append(max(multitoken_result))

            multitoken_result = []
        elif t.startswith("▁"):
            if len(multitoken_result) > 0:
                # p_result.append(find_majority_element(multitoken_result))
                p_result.append(max(multitoken_result))
            multitoken_result = [p]

        else:
            multitoken_result.append(p)

    # add final multitoken result
    if len(multitoken_result) > 0:
        # p_result.append(find_majority_element(multitoken_result))
        p_result.append(max(multitoken_result))

    return p_result


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        ptokens_bos, _ = batch.ptokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # forward modules
        w2v_out = self.modules.SSL_enc(wavs)

        ## ASR head ##
        out_ASR, _ = self.modules.Transformer(
            w2v_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(w2v_out)
        p_ctc = self.hparams.log_softmax(logits)
        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(out_ASR)
        p_seq = self.hparams.log_softmax(pred)

        p_para = self.modules.para_seq_lin(out_ASR)
        para_seq = self.hparams.log_softmax(p_para)

        # Compute outputs
        hyps_para = None
        hyps_asr = None
        hyps_attn = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                hyps_asr, _, hyps_para = self.hparams.valid_search(
                    w2v_out.detach(), wav_lens
                )

        elif stage == sb.Stage.TEST:
            hyps_asr, _, hyps_para = self.hparams.test_search(
                w2v_out.detach(), wav_lens
            )

        return p_ctc, p_seq, para_seq, wav_lens, hyps_para, hyps_asr, hyps_attn

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        (
            p_ctc,
            p_seq,
            para_seq,
            wav_lens,
            hyps_para,
            hyps_asr,
            hyps_attn,
        ) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        ptokens_eos, ptokens_eos_lens = batch.ptokens_eos
        tokens, tokens_lens = batch.tokens
        ptokens, ptokens_lens = batch.ptokens

        # ASR
        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()
        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        # paraphasia loss
        para_index = 1
        weight_arr = torch.ones(2)
        weight_arr[para_index] = 2
        loss_para_seq = self.hparams.seq_cost(
            # para_seq, ptokens_eos, length=ptokens_lens, weight=weight_arr.to(self.device)
            para_seq,
            ptokens_eos,
            length=ptokens_lens,
        ).sum()
        # Use ptokens_lens to mask out eos. Use ptokens_eos as labels to match predictions, eos will be masked

        loss_asr = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        # loss scaling
        if hasattr(self, "loss_scaler"):
            self.loss_scaler.update_avg_losses(loss_asr, loss_para_seq)
            scaled_loss_asr, scaled_loss_para = self.loss_scaler.scale_losses(
                loss_asr, loss_para_seq
            )
            loss = scaled_loss_asr + scaled_loss_para
        else:
            loss = loss_asr, loss_para_seq

        # print(f"pre asr: {loss_asr} | para: {loss_para_seq}")
        # print(f"scaled asr: {scaled_loss_asr} | para: {scaled_loss_para}")

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):

                predicted_words = [
                    self.tokenizer.decode_ids(utt_seq).split(" ")
                    for utt_seq in hyps_asr
                ]
                predicted_words = [
                    [p for p in predicted_words[0] if p != ""]
                ]  # ensure padding gets trimmed

                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.asr_wer_metric.append(ids, predicted_words, target_words)
                self.asr_cer_metric.append(ids, predicted_words, target_words)

                # combine token-level paraphasia for AWER
                pred_AWER = combine_AWER(
                    hyps_para[0],
                    hyps_asr[0],
                    predicted_words[0],
                    self.tokenizer,
                    self.ptokenizer,
                )
                target_AWER = [wrd.split(" ") for wrd in batch.aug_para]
                self.para_wer_metric.append(ids, pred_AWER, target_AWER)

                # Baseline AWER
                baseline_AWER = [
                    [f"{p}/C" for p_utt in predicted_words for p in p_utt]
                ]
                self.baseline_awer_metric.append(
                    ids, baseline_AWER, target_AWER
                )

            # compute the accuracy of the one-step-forward prediction
            self.asr_acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
            self.para_acc_metric.append(para_seq, ptokens_eos, ptokens_eos_lens)
        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        # print(f"LR: {self.optimizer.param_groups[-1]['lr']}")
        if stage != sb.Stage.TRAIN:
            self.para_wer_metric = self.hparams.error_rate_computer()
            self.baseline_awer_metric = self.hparams.error_rate_computer()
            self.asr_cer_metric = self.hparams.cer_computer()
            self.asr_wer_metric = self.hparams.error_rate_computer()

            self.asr_acc_metric = self.hparams.acc_computer()
            self.para_acc_metric = self.hparams.acc_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch.
        If DDP only for test and train"""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            # multi-gpu decoding
            if self.distributed_launch and stage == sb.Stage.TEST:
                # stage_stats["ACC"] = self.acc_metric.summarize_dist(self.device)
                stage_stats["asr-ACC"] = self.asr_acc_metric.summarize_dist(
                    self.device
                )
                stage_stats["p-ACC"] = self.para_acc_metric.summarize_dist(
                    self.device
                )

                # stage_stats["WER"] = self.wer_metric.summarize_dist("WER","error_rate")
                # stage_stats["CER"] = self.cer_metric.summarize_dist("CER","error_rate")
                stage_stats["asr-WER"] = self.asr_wer_metric.summarize_dist(
                    "WER", "error_rate"
                )
                stage_stats["asr-CER"] = self.asr_cer_metric.summarize_dist(
                    "CER", "error_rate"
                )
                stage_stats["p-AWER"] = self.para_wer_metric.summarize_dist(
                    "P-AWER", "error_rate"
                )
                stage_stats[
                    "baseline-AWER"
                ] = self.baseline_awer_metric.summarize_dist(
                    "B-AWER", "error_rate"
                )
            else:
                stage_stats["asr-ACC"] = self.asr_acc_metric.summarize()
                stage_stats["p-ACC"] = self.para_acc_metric.summarize()

            # create variable to track
            self.stage_stats = stage_stats

            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0:
                stage_stats["asr-WER"] = self.asr_wer_metric.summarize(
                    "error_rate"
                )
                stage_stats["asr-CER"] = self.asr_cer_metric.summarize(
                    "error_rate"
                )
                stage_stats["p-AWER"] = self.para_wer_metric.summarize(
                    "error_rate"
                )

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            # lr = self.hparams.noam_annealing.current_lr
            # newBOB
            lr, new_lr_model = self.hparams.lr_annealing(stage_stats["loss"])
            sb.nnet.schedulers.update_learning_rate(
                self.optimizer, new_lr_model
            )

            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_stats["loss"], "epoch": epoch},
                min_keys=["loss"],
                num_to_keep=1,
            )

        elif stage == sb.Stage.TEST:
            if sb.utils.distributed.if_main_process():
                self.hparams.train_logger.log_stats(
                    stats_meta={
                        "Epoch loaded": self.hparams.epoch_counter.current
                    },
                    test_stats=stage_stats,
                )

                with open(self.hparams.asr_wer_file, "w") as w:
                    self.asr_wer_metric.write_stats(w)

                with open(self.hparams.asr_cer_file, "w") as w:
                    self.asr_cer_metric.write_stats(w)

                with open(self.hparams.para_awer_file, "w") as w:
                    self.para_wer_metric.write_stats(w)

                with open(self.hparams.baseline_awer_file, "w") as w:
                    self.baseline_awer_metric.write_stats(w)

    def fit_batch(self, batch):
        should_step = self.step % self.grad_accumulation_factor == 0
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            with torch.cuda.amp.autocast():
                outputs = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                self.scaler.scale(
                    loss / self.grad_accumulation_factor
                ).backward()
            if should_step:
                self.scaler.unscale_(self.optimizer)
                if self.check_gradients(loss):
                    self.scaler.step(self.optimizer)
                self.scaler.update()
                self.zero_grad()
                self.optimizer_step += 1
                # self.hparams.noam_annealing(self.optimizer)
        else:
            outputs = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
            with self.no_sync(not should_step):
                (loss / self.grad_accumulation_factor).backward()
            if should_step:
                if self.check_gradients(loss):
                    self.optimizer.step()
                self.zero_grad()
                self.optimizer_step += 1
                # self.hparams.noam_annealing(self.optimizer)

        self.on_fit_batch_end(batch, outputs, loss, should_step)
        return loss.detach().cpu()

    def on_fit_start(self):
        """Gets called at the beginning of ``fit()``, on multiple processes
        if ``distributed_count > 0`` and backend is ddp.

        Default implementation compiles the jit modules, initializes
        optimizers, and loads the latest checkpoint to resume training.
        """
        # Run this *after* starting all processes since jit modules cannot be
        # pickled.
        self._compile_jit()

        # Wrap modules with parallel backend after jit
        self._wrap_distributed()

        # Initialize optimizers after parameters are configured
        self.init_optimizers()

        # Load latest checkpoint to resume training if interrupted
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                device=torch.device(self.device)
            )

        if self.hparams.FT_start:
            old_lr = self.hparams.lr_annealing.hyperparam_value
            self.hparams.lr_annealing.hyperparam_value = self.hparams.lr_adam
            print(
                f"restart lr: {old_lr} -> {self.hparams.lr_annealing.hyperparam_value}"
            )

    def make_dataloader(
        self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs
    ):
        # TRAIN stage is handled specially.
        # if stage == sb.Stage.TRAIN:
        #     loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
        if stage == sb.Stage.TRAIN or stage == sb.Stage.TEST:
            loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)
        # loader_kwargs = self._train_loader_specifics(dataset, loader_kwargs)

        dataloader = sb.dataio.dataloader.make_dataloader(
            dataset, **loader_kwargs
        )

        if (
            self.checkpointer is not None
            and ckpt_prefix is not None
            and (
                isinstance(dataloader, SaveableDataLoader)
                or isinstance(dataloader, LoopedLoader)
            )
        ):
            ckpt_key = ckpt_prefix + stage.name
            self.checkpointer.add_recoverable(ckpt_key, dataloader)
        return dataloader

    def evaluate(
        self,
        test_set,
        max_key=None,
        min_key=None,
        progressbar=None,
        test_loader_kwargs={},
    ):
        if progressbar is None:
            progressbar = not self.noprogressbar

        if not (
            isinstance(test_set, DataLoader)
            or isinstance(test_set, LoopedLoader)
        ):
            test_loader_kwargs["ckpt_prefix"] = None
            test_set = self.make_dataloader(
                test_set, sb.Stage.TEST, **test_loader_kwargs
            )
        self.on_evaluate_start(max_key=max_key, min_key=min_key)
        self.on_stage_start(sb.Stage.TEST, epoch=None)
        self.modules.eval()
        avg_test_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(
                test_set,
                dynamic_ncols=True,
                disable=not progressbar,
                colour=self.tqdm_barcolor["test"],
            ):
                self.step += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.TEST)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # on every process
            self.on_stage_end(sb.Stage.TEST, avg_test_loss, None)
        self.step = 0
        return avg_test_loss

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs={},
        valid_loader_kwargs={},
    ):
        if not (
            isinstance(train_set, DataLoader)
            or isinstance(train_set, LoopedLoader)
        ):
            train_set = self.make_dataloader(
                train_set, stage=sb.Stage.TRAIN, **train_loader_kwargs
            )
        if valid_set is not None and not (
            isinstance(valid_set, DataLoader)
            or isinstance(valid_set, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()

        if progressbar is None:
            progressbar = not self.noprogressbar

        # Only show progressbar if requested and main_process
        enable = progressbar and sb.utils.distributed.if_main_process()

        # Reset epoch_counter for FT
        if self.hparams.FT_start and isinstance(
            epoch_counter, sb.utils.epoch_loop.EpochCounterWithStopper
        ):
            print(
                f"epoch_counter:\n curr: {epoch_counter.current}\n"
                f"limit: {epoch_counter.limit}\n"
                f"last_best_epoch: {epoch_counter.best_limit}\n"
                f"limit_to_stop: {epoch_counter.limit_to_stop}\n"
                f"limit_warmup: {epoch_counter.limit_warmup}\n"
                f"th: {epoch_counter.th}"
            )
            # # since its being FT, reset with current
            # self.epoch_counter_limit_to_stop_FT = epoch_counter.limit_to_stop
            # self.epoch_counter_limit_warmup_FT = epoch_counter.limit_warmup
            # epoch_counter.limit_to_stop = epoch_counter.current + self.epoch_counter_limit_to_stop_FT
            # epoch_counter.limit_warmup = epoch_counter.current + self.epoch_counter_limit_warmup_FT

            self.epoch_counter_limit_warmup_FT = epoch_counter.limit_warmup
            epoch_counter.limit_warmup = (
                epoch_counter.current + self.epoch_counter_limit_warmup_FT
            )

            print(
                f"POST epoch_counter:\n curr: {epoch_counter.current}\n"
                f"limit: {epoch_counter.limit}\n"
                f"last_best_epoch: {epoch_counter.best_limit}\n"
                f"patience: {epoch_counter.current - epoch_counter.best_limit}\n"
                f"limit_to_stop: {epoch_counter.limit_to_stop}\n"
                f"limit_warmup: {epoch_counter.limit_warmup}\n"
                f"th: {epoch_counter.th}"
            )

        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            if isinstance(
                epoch_counter, sb.utils.epoch_loop.EpochCounterWithStopper
            ) and epoch_counter.should_stop(
                current=epoch, current_metric=self.valid_loss
            ):
                epoch_counter.current = epoch_counter.limit

            print(
                f"POST epoch_counter:\n curr: {epoch_counter.current}\n"
                f"limit: {epoch_counter.limit}\n"
                f"last_best_epoch: {epoch_counter.best_limit}\n"
                f"patience: {epoch_counter.current - epoch_counter.best_limit}\n"
                f"limit_to_stop: {epoch_counter.limit_to_stop}\n"
                f"limit_warmup: {epoch_counter.limit_warmup}\n"
                f"th: {epoch_counter.th}"
            )

            # tensorboard
            self.tb.add_scalar("train_loss", self.tb_avg_train_loss, epoch)
            self.tb.add_scalar("valid_loss", self.valid_loss, epoch)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break

    def _fit_valid(self, valid_set, epoch, enable):
        # Validation stage
        if valid_set is not None:
            self.on_stage_start(sb.Stage.VALID, epoch)
            self.modules.eval()
            avg_valid_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(
                    valid_set,
                    dynamic_ncols=True,
                    disable=not enable,
                    colour=self.tqdm_barcolor["valid"],
                ):
                    self.step += 1
                    loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                    avg_valid_loss = self.update_average(loss, avg_valid_loss)

                    # Profile only if desired (steps allow the profiler to know when all is warmed up)
                    if self.profiler is not None:
                        if self.profiler.record_steps:
                            self.profiler.step()

                    # Debug mode only runs a few batches
                    if self.debug and self.step == self.debug_batches:
                        break

                # Only run validation "on_stage_end" on main process
                self.step = 0
                self.valid_loss = (
                    avg_valid_loss  # needed for epoch_counter_Stop
                )
                run_on_main(
                    self.on_stage_end,
                    args=[sb.Stage.VALID, avg_valid_loss, epoch],
                )

    def _fit_train(self, train_set, epoch, enable):
        # Training stage
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()

        # Reset nonfinite count to 0 each epoch
        self.nonfinite_count = 0

        if self.train_sampler is not None and hasattr(
            self.train_sampler, "set_epoch"
        ):
            self.train_sampler.set_epoch(epoch)

        # Time since last intra-epoch checkpoint
        last_ckpt_time = time.time()
        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as t:
            for batch in t:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                t.set_postfix(train_loss=self.avg_train_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

                if (
                    self.checkpointer is not None
                    and self.ckpt_interval_minutes > 0
                    and time.time() - last_ckpt_time
                    >= self.ckpt_interval_minutes * 60.0
                ):
                    # This should not use run_on_main, because that
                    # includes a DDP barrier. That eventually leads to a
                    # crash when the processes'
                    # time.time() - last_ckpt_time differ and some
                    # processes enter this block while others don't,
                    # missing the barrier.
                    if sb.utils.distributed.if_main_process():
                        self._save_intra_epoch_ckpt()
                    last_ckpt_time = time.time()

        self.tb_avg_train_loss = self.avg_train_loss
        # Run train "on_stage_end" on all processes
        self.zero_grad(set_to_none=True)  # flush gradients
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0
        self.step = 0


def align_labels_to_tokenized_input(wrd, labels, tokenizer, paraphasia_dict):
    words = wrd.split()
    labels = labels.split()
    assert len(words) == len(
        labels
    ), "Sentences and labels lists should be of same size."

    return_labels = []
    return_tokens = []
    for w, l in zip(words, labels):
        # print(f"w: {w} | l: {l}")
        tokenized_word = tokenizer.sp.encode(w, out_type=str)
        tokenized_label = [paraphasia_dict[l]] * len(tokenized_word)

        return_labels.extend(tokenized_label)
        return_tokens.extend(tokenized_word)

    return return_tokens, return_labels


def dataio_prepare(hparams, tokenizer, ptoknizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            key_max_value={"duration": hparams["max_length"]},
            # key_max_value={"duration": 0.8},
            key_min_value={"duration": hparams["min_length"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration",
            reverse=True,
            key_max_value={"duration": hparams["max_length"]},
            # key_max_value={"duration": 0.8},
            key_min_value={"duration": hparams["min_length"]},
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder}
    )
    valid_data = valid_data.filtered_sorted(
        sort_key="duration",
        reverse=True,
        key_max_value={"duration": hparams["max_length"]},
        # key_max_value={"duration": 0.8},
        key_min_value={"duration": hparams["min_length"]},
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    test_data = test_data.filtered_sorted(
        sort_key="duration",
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]},
    )

    datasets = [train_data, valid_data, test_data]
    valtest_datasets = [valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if hparams["speed_perturb"]:
            sig = sb.dataio.dataio.read_audio(wav)
            # factor = np.random.uniform(0.95, 1.05)
            # sig = resample(sig.numpy(), 16000, int(16000*factor))
            speed = sb.processing.speech_augmentation.SpeedPerturb(
                16000, [x for x in range(95, 105)]
            )
            sig = speed(sig.unsqueeze(0)).squeeze(0)  # torch.from_numpy(sig)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd", "aug_para", "paraphasia", "id")
    @sb.utils.data_pipeline.provides(
        "wrd",
        "tokens_list",
        "tokens_bos",
        "tokens_eos",
        "tokens",
        "aug_para",
        "paraphasia_word_level",
        "ptokens_bos",
        "ptokens_eos",
        "ptokens",
    )
    def text_pipeline(wrd, aug_para, paraphasia, id):
        yield wrd
        # align paraphasia labels to tokenized text output
        _, ptokens_list = align_labels_to_tokenized_input(
            wrd, paraphasia, tokenizer, ptokenizer
        )
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

        # paraphasia
        yield aug_para
        paraphasia_word_level = [ptokenizer[p] for p in paraphasia.split()]
        yield paraphasia_word_level
        ptokens_bos = torch.LongTensor([hparams["bos_index"]] + (ptokens_list))
        yield ptokens_bos
        ptokens_eos = torch.LongTensor(ptokens_list + [0])
        yield ptokens_eos
        ptokens = torch.LongTensor(ptokens_list)
        yield ptokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        [
            "id",
            "sig",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "aug_para",
            "paraphasia_word_level",
            "ptokens_bos",
            "ptokens_eos",
            "ptokens",
        ],
    )

    return (
        train_data,
        valid_data,
        test_data,
    )


def prep_exp_dir(hparams):
    save_folder = hparams["save_folder"]
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f"run_opts: {run_opts}")
    # exit()

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    prep_exp_dir(hparams)

    # Defining tokenizer and loading it
    tokenizer = SentencePiece(
        model_dir=hparams["save_folder"],
        vocab_size=hparams["output_neurons"],
        annotation_train=hparams["train_csv"],
        annotation_read="wrd",
        model_type=hparams["token_type"],
        character_coverage=1.0,
        bos_id=hparams["bos_index"],
        eos_id=hparams["eos_index"],
        pad_id=hparams["pad_index"],
        unk_id=hparams["unk_index"],
    )
    ptokenizer = {
        "c": 0,
        "p": 1,
        "n": 2,
        "s": 3,
    }
    reverse_ptokenizer = ["c", "p", "n", "s"]

    train_data, valid_data, test_data = dataio_prepare(
        hparams, tokenizer, ptokenizer
    )

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    asr_brain.tokenizer = tokenizer.sp
    asr_brain.ptokenizer = reverse_ptokenizer
    tokens = {
        i: asr_brain.tokenizer.id_to_piece(i)
        for i in range(asr_brain.tokenizer.get_piece_size())
    }

    if sb.utils.distributed.if_main_process():
        # print(f"tokenizer: {tokens} | {len(tokens.keys())}")
        count_parameters(asr_brain.modules)

    asr_brain.tb = SummaryWriter(hparams["tb_logs"])

    if "dynamic_loss_scaler_alpha" in hparams:
        alpha_val = 0.8
        asr_brain.loss_scaler = DynamicLossScaler(alpha=alpha_val)

    # with torch.autograd.detect_anomaly():
    if hparams["train_flag"]:
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )

    print("\nEVALUATE\n")
    # Testing

    asr_brain.hparams.asr_wer_file = os.path.join(
        hparams["output_folder"], "asr_wer.txt"
    )
    asr_brain.hparams.para_awer_file = os.path.join(
        hparams["output_folder"], "awer_para.txt"
    )
    asr_brain.hparams.baseline_awer_file = os.path.join(
        hparams["output_folder"], "awer_baseline.txt"
    )
    asr_brain.hparams.asr_cer_file = os.path.join(
        hparams["output_folder"], "asr_cer.txt"
    )
    asr_brain.hparams.para_f1_file = os.path.join(
        hparams["output_folder"], "para_f1.txt"
    )

    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )
