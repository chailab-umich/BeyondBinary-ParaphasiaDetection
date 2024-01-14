#!/usr/bin/env/python3
"""
S2S with Transformer-encoder and decoder.

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
from speechbrain.utils.distributed import run_on_main, if_main_process
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
from datasets import load_dataset, load_metric, Audio
import re
import time
from speechbrain.tokenizers.SentencePiece import SentencePiece
#multi-gpu
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.dataio.dataloader import SaveableDataLoader
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm
import gc


logger = logging.getLogger(__name__)

torch.autograd.set_detect_anomaly(True)

def props(cls):   
    return [i for i in cls.__dict__.keys() if i[:1] != '_']
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        wavs, wav_lens = wavs.to(self.device), wav_lens.to(self.device)

        # Add augmentation if specified
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, wav_lens)

        # forward modules
        w2v_out = self.modules.SSL_enc(wavs)

        ## ASR head ##
        out_ASR, attn = self.modules.Transformer(
            w2v_out, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )
        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(w2v_out)
        p_ctc = self.hparams.log_softmax(logits)
        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(out_ASR)
        p_seq = self.hparams.log_softmax(pred)


        # Compute outputs
        hyps_asr = None
        hyps_attn = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            current_epoch = self.hparams.epoch_counter.current
            if (current_epoch % self.hparams.valid_search_interval == 0):
                hyps_asr, _ = self.hparams.valid_search(w2v_out.detach(), wav_lens)

        elif stage == sb.Stage.TEST:
            hyps_asr, _ = self.hparams.test_search(w2v_out.detach(), wav_lens)

        return p_ctc, p_seq, wav_lens, hyps_asr, hyps_attn
    
    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        (p_ctc, p_seq, wav_lens, hyps, hyps_attn) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        # now as training progresses we use real prediction from the prev step instead of teacher forcing

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval

            if current_epoch % valid_search_interval == 0 or stage == sb.Stage.TEST:
            
                predicted_words = [
                    self.tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd] # AB

                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)



            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        with torch.no_grad():
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.acc_metric = self.hparams.acc_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")
                stage_stats["CER"] = self.cer_metric.summarize("error_rate")

        # create variable to track
        self.stage_stats = stage_stats
        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # lr = self.hparams.noam_annealing.current_lr
            # newBOB
            lr, new_lr_model = self.hparams.lr_annealing(
                stage_stats["loss"]
            )
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
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

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

    def make_dataloader(self, dataset, stage, ckpt_prefix="dataloader-", **loader_kwargs):
        # TRAIN stage is handled specially.
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
        stage=sb.Stage.TEST,
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
                test_set, stage, **test_loader_kwargs
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
                loss = self.evaluate_batch(batch, stage=stage)
                avg_test_loss = self.update_average(loss, avg_test_loss)

                # Profile only if desired (steps allow the profiler to know when all is warmed up)
                if self.profiler is not None:
                    if self.profiler.record_steps:
                        self.profiler.step()

                # Debug mode only runs a few batches
                if self.debug and self.step == self.debug_batches:
                    break

            # # Only run evaluation "on_stage_end" on main process
            # run_on_main(
            #     self.on_stage_end, args=[Stage.TEST, avg_test_loss, None]
            # )
            self.on_stage_end(stage, avg_test_loss, None)

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



        # Iterate epochs
        for epoch in epoch_counter:
            self._fit_train(train_set=train_set, epoch=epoch, enable=enable)
            self._fit_valid(valid_set=valid_set, epoch=epoch, enable=enable)

            epoch_counter.update_metric(self.valid_loss)

            # Debug mode only runs a few epochs
            if (
                self.debug
                and epoch == self.debug_epochs
                or self._optimizer_step_limit_exceeded
            ):
                break


def dataio_prepare(hparams,tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    #convert severity_cat to int
    train_data.data = {k:{k_2: (int(v_2) if k_2 == 'severity_cat' else v_2) for k_2,v_2 in v.items()} for k,v in train_data.data.items()}
    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration",
            key_max_value={"duration": hparams["max_length"], "severity_cat": hparams["max_sev_train"]},
            # key_max_value={"duration": 1, "severity_cat": hparams["max_sev_train"]},
            key_min_value={"duration": hparams["min_length"], "severity_cat": hparams["min_sev_train"]},

        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False
    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True,
            # key_max_value={"duration": hparams["max_length"]}
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder}
    )
    
    valid_data = valid_data.filtered_sorted(sort_key="duration",
        key_max_value={"duration": hparams["val_max_length"]},
        key_min_value={"duration": hparams["val_min_length"]}
    )


    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder}
    )
    test_data = test_data.filtered_sorted(sort_key="duration",
        # key_max_value={"duration": 1},
        key_max_value={"duration": hparams["max_length"]},
        key_min_value={"duration": hparams["min_length"]}
    )

    datasets = [train_data, valid_data, test_data]
    valtest_datasets = [valid_data,test_data]

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
    @sb.utils.data_pipeline.takes("wrd","aug_para")
    @sb.utils.data_pipeline.provides(
        "wrd", "transcription_para", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd,aug_para):
        yield wrd
        transcription_para = aug_para.replace("/c","")
        transcription_para = transcription_para.replace("/n","[n]")
        transcription_para = transcription_para.replace("/p","[p]")
        yield transcription_para
        tokens_list = tokenizer.sp.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "transcription_para", "tokens_bos", "tokens_eos", "tokens"],
    )


    return (
        train_data,
        valid_data,
        test_data
    )


def prep_exp_dir(hparams):
    save_folder = hparams['save_folder']
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print(f"run_opts: {run_opts}")

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
        user_defined_symbols=hparams["user_defined_symbols"],
    )
    # vocab_check(tokenizer, hparams)



    train_data,valid_data,test_data = dataio_prepare(hparams,tokenizer=tokenizer)

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )


    asr_brain.tokenizer = tokenizer.sp
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]
    tokens = {i:asr_brain.tokenizer.id_to_piece(i) for i in range(asr_brain.tokenizer.get_piece_size())}
    print(f"tokenizer: {tokens} | {len(tokens.keys())}")
    # exit()
    
    # asr_brain.modules = asr_brain.modules.float()
    count_parameters(asr_brain.modules)


    with torch.autograd.detect_anomaly():
        if hparams['train_flag']:
            print("training model")
            asr_brain.fit(
                asr_brain.hparams.epoch_counter,
                train_data,
                valid_data,
                train_loader_kwargs=hparams["train_dataloader_opts"],
                valid_loader_kwargs=hparams["valid_dataloader_opts"],
            )


    # Testing
    print("Run Eval")
    asr_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer.txt"
    )
    asr_brain.hparams.cer_file = os.path.join(
        hparams["output_folder"], "cer.txt"
    )


     ## LM sweep ##
    print("LM SWEEP")
    best_wer = float('inf')
    best_ctc = torch.Tensor([0.0]).cuda()
    best_lm = torch.Tensor([0.0]).cuda()
    for ctc_weight in [0.2,0.3,0.4]:
        for lm_weight in [0.0]:
            asr_brain.hparams.test_search.ctc_weight = ctc_weight
            asr_brain.hparams.test_search.lm_weight = lm_weight

            asr_brain.evaluate(
                valid_data, test_loader_kwargs=hparams["valid_dataloader_opts"]
            )
            # print(f"stage_stats: {asr_brain.stage_stats} | {os.environ['RANK']}")
            if if_main_process():
                val_wer = asr_brain.stage_stats['WER']
                if val_wer < best_wer:
                    best_wer = val_wer
                    best_ctc = torch.Tensor([ctc_weight]).cuda()
                    best_lm = torch.Tensor([lm_weight]).cuda()
                print(f"ctc: {asr_brain.hparams.test_search.ctc_weight} | lm: {asr_brain.hparams.test_search.lm_weight} | wer: {val_wer} | best_wer: {best_wer}")
            torch.cuda.empty_cache()

    # share variables to other models
    torch.distributed.broadcast(best_ctc, src=0)
    torch.distributed.broadcast(best_lm, src=0)
    best_ctc = round(best_ctc.item(),1)
    best_lm = round(best_lm.item(),1)

    asr_brain.hparams.test_search.ctc_weight = best_ctc
    asr_brain.hparams.test_search.lm_weight = best_lm
    print(f"EVALUATE best test_search params lm: {best_lm} | ctc: {best_ctc} | {os.environ['RANK']}")

    asr_brain.evaluate(
        test_data, test_loader_kwargs=hparams["test_dataloader_opts"]
    )