import random
import logging
from typing import Dict, OrderedDict
from transformers import Trainer
import torch
import torch.nn as nn
from transformers import BertModel

import matplotlib.pyplot as plt

import time
from typing import List, Optional, Union
from torch.utils.data import Dataset
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption
import math

from utils.focal_loss import focal_loss
from utils.multi_label import multilabel_categorical_crossentropy, sparse_multilabel_categorical_crossentropy
from utils.utils_int import random_int_list, minimum_int_list

from torch.utils.tensorboard import SummaryWriter

from transformers.file_utils import is_sagemaker_mp_enabled

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import (
        smp_forward_backward,
        smp_forward_only,
        smp_gather,
        smp_nested_concat,
    )

from typing import Dict, OrderedDict, Union, Any



logger = logging.getLogger(__name__)

_default_log_level = logging.INFO
logger.setLevel(_default_log_level)


class IntentTrainer(Trainer):
    def __init__(
        self,
        intent_label_set,
        config,
        model_args,
        predict_dataset=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.test_key = "f1score"
        self.model_args = model_args
        self.best_metrics = OrderedDict(
            {
                "best_epoch": 0,
                f"best_eval_{self.test_key}": 0,
            }
        )

        self.predict_dataset = predict_dataset
        self.epoch = 0

        self.config = config
        self.bert = BertModel(config)
        self.intent_label_set = intent_label_set
            # classifier_dropout = (
            #     config.classifier_dropout
            #     if config.classifier_dropout is not None
            #     else config.hidden_dropout_prob
            # )
        try:
            self.dropout = nn.Dropout(config.classifier_dropout)
        except:
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, config.num_labels).to(
            self.model.device
        )
        self.sigmoid = nn.Sigmoid().to(self.model.device)

        self.writer = SummaryWriter(
            f"./tensorboard_log/{self.model_args.experiment_name}"
        )
        fake_input = torch.randn(config.hidden_size).to(self.model.device)
        self.writer.add_graph(model=self.classifier, input_to_model=fake_input)

    def log_best_metrics(self):
        self.log_metrics("best", self.best_metrics)
        self.save_metrics("best", self.best_metrics, combined=False)

    def _maybe_log_save_evaluate(
        self, tr_loss, model, trial, epoch, ignore_keys_for_eval
    ):
        self.epoch = epoch
        if self.control.should_log:
            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        eval_metrics = None
        if self.control.should_evaluate:

            if "train" in self.model_args.eval_type:
                logger.info(f"***** Running Evaluation for train dataset *****")
                train_metrics, train_predictions, train_labels = self.evaluate(
                    ignore_keys=ignore_keys_for_eval,
                    eval_dataset=self.train_dataset,
                    metric_key_prefix="train",
                )
                self._report_to_hp_search(trial, epoch, train_metrics)

            if "test" in self.model_args.eval_type:
                logger.info(f"***** Running Evaluation for test dataset *****")
                test_metrics, test_predictions, test_labels = self.evaluate(
                    ignore_keys=ignore_keys_for_eval,
                    eval_dataset=self.predict_dataset,
                    metric_key_prefix="test",
                )
                self._report_to_hp_search(trial, epoch, train_metrics)

            logger.info(f"***** Running Evaluation for eval dataset *****")
            eval_metrics, eval_predictions, eval_labels = self.evaluate(
                ignore_keys=ignore_keys_for_eval, metric_key_prefix="eval"
            )
            self._report_to_hp_search(trial, epoch, eval_metrics)

            for key in [
                "RocAucScore",
                "ExactMatchRatio",
                "ZeroOneLoss",
                "accuracy",
                "precision",
                "precision",
                "recall",
                "f1score",
                "hammingloss",
                "positive_number_per_sample",
                "positive_rate",
            ]:
                self.writer.add_scalars(
                    f"metrics/{key}",
                    {
                        "train": train_metrics[f"train_{key}"],
                        "eval": eval_metrics[f"eval_{key}"],
                        "test": test_metrics[f"test_{key}"],
                    },
                    epoch,
                )

            for key in ["loss"]:
                self.writer.add_scalars(
                    f"loss/{key}",
                    {
                        "train": train_metrics[f"train_{key}"],
                        "eval": eval_metrics[f"eval_{key}"],
                        "test": test_metrics[f"test_{key}"],
                    },
                    epoch,
                )

            self.writer.add_histogram(
                tag="train_predictions", values=train_predictions, global_step=epoch
            )
            self.writer.add_histogram(
                tag="eval_predictions", values=eval_predictions, global_step=epoch
            )
            self.writer.add_histogram(
                tag="test_predictions", values=test_predictions, global_step=epoch
            )

            # self.writer.add_embedding(
            #     tag="train_prediction",
            #     mat=train_predictions,
            #     metadata=self.train_dataset["label_index"],
            #     global_step=epoch,
            # )
            # self.writer.add_embedding(
            #     tag="eval_prediction",
            #     mat=eval_predictions,
            #     metadata=self.eval_dataset["label_index"],
            #     global_step=epoch,
            # )
            # self.writer.add_embedding(
            #     tag="test_prediction",
            #     mat=test_predictions,
            #     metadata=self.predict_dataset["label_index"],
            #     global_step=epoch,
            # )

            train_number_ones, eval_number_ones, test_number_ones = {}, {}, {}
            for index in range(0, len(self.intent_label_set), 1):
                train_number_ones[str(index)] = 0
                eval_number_ones[str(index)] = 0
                test_number_ones[str(index)] = 0
            for row in range(len(train_predictions)):
                num_train_ones = (train_predictions[row] >= 0.5).sum()
                train_number_ones[str(num_train_ones)] += 1
            for row in range(len(eval_predictions)):
                num_eval_ones = (eval_predictions[row] >= 0.5).sum()
                eval_number_ones[str(num_eval_ones)] += 1
            for row in range(len(test_predictions)):
                num_test_ones = (test_predictions[row] >= 0.5).sum()
                test_number_ones[str(num_test_ones)] += 1
            # for number_ones, num_samples in {train_number_ones: len(train_predictions), eval_number_ones: len(eval_predictions), test_number_ones: len(test_predictions)}.items():
            #     for key in number_ones.keys():
            #         number_ones[key] /= num_samples

            fig = plt.figure()
            plt.subplot(221)
            x_train = []
            y_train = []
            for key, value in train_number_ones.items():
                x_train.append(int(key))
                y_train.append(value)
            plt.bar(x_train, y_train, width=0.5, facecolor="lightblue")
            plt.title("number of ones at training")

            plt.subplot(222)
            x_eval = []
            y_eval = []
            for key, value in eval_number_ones.items():
                x_eval.append(int(key))
                y_eval.append(value)
            plt.bar(x_eval, y_eval, width=0.5, facecolor="lightblue")
            plt.title("number of ones at eval")

            plt.subplot(223)
            x_test = []
            y_test = []
            for key, value in test_number_ones.items():
                x_test.append(int(key))
                y_test.append(value)
            plt.bar(x_test, y_test, width=0.5, facecolor="lightblue")
            plt.title("number of ones at test")

            self.writer.add_figure(
                tag=f"numbe_of_ones_{str(epoch)}", figure=fig, global_step=epoch
            )

            # PR_Curve
            self.writer.add_pr_curve(f'PR_Curve_train_{epoch}', train_labels, train_predictions, epoch)
            self.writer.add_pr_curve(f'PR_Curve_eval_{epoch}', eval_labels, eval_predictions, epoch)
            self.writer.add_pr_curve(f'PR_Curve_test_{epoch}', test_labels, test_predictions, epoch)

            # HParameters
            set_metrics = {
                "train": train_metrics,
                "eval": eval_metrics,
                "test": test_metrics
            }
            for key, value in set_metrics.items():
                self.writer.add_hparams(
                    {
                        "lr": self.args.learning_rate,
                        "batch_size": self.args.per_device_train_batch_size,
                        "epoch": epoch,
                        "split_set": key
                    },
                    {
                        "loss": value[f"{key}_loss"],
                        "precision": value[f"{key}_precision"],
                        "recall": value[f"{key}_recall"],
                        "f1score": value[f"{key}_f1score"],
                        "RocAucScore": value[f"{key}_RocAucScore"],
                        "ExactMatchRatio": value[f"{key}_ExactMatchRatio"],
                        "ZeroOneLoss": value[f"{key}_ZeroOneLoss"],
                        "accuracy": value[f"{key}_accuracy"],
                        "hammingloss": value[f"{key}_hammingloss"],
                        "positive_number_per_sample": value[f"{key}_positive_number_per_sample"],
                        "positive_rate": value[f"{key}_positive_rate"]
                    },
                )

            if (
                eval_metrics["eval_" + self.test_key]
                > self.best_metrics["best_eval_" + self.test_key]
            ):
                self.best_metrics["best_epoch"] = epoch
                self.best_metrics["best_eval_" + self.test_key] = eval_metrics[
                    "eval_" + self.test_key
                ]

                if self.predict_dataset is not None:
                    if isinstance(self.predict_dataset, dict):
                        for dataset_name, dataset in self.predict_dataset.items():
                            _, _, test_metrics = self.predict(
                                dataset, metric_key_prefix="test"
                            )
                            self.best_metrics[
                                f"best_test_{dataset_name}_{self.test_key}"
                            ] = test_metrics["test_" + self.test_key]
                    else:
                        _, _, test_metrics = self.predict(
                            self.predict_dataset, metric_key_prefix="test"
                        )
                        self.best_metrics["best_test_" + self.test_key] = test_metrics[
                            "test_" + self.test_key
                        ]

            logger.info(f"***** Epoch {epoch}: Best results *****")
            for key, value in self.best_metrics.items():
                logger.info(f"{key} = {value}")
            self.log(self.best_metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=eval_metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = (
            self.prediction_loop
            if self.args.use_legacy_prediction_loop
            else self.evaluation_loop
        )
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, output.metrics
        )

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics, output.predictions, output.label_ids

    def _prepare_inputs(
        self, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                f"training dataset contains keys expected by the model: {','.join(self._signature_columns)}."
            )
        if self.args.past_index >= 0 and self._past is not None:
            inputs["mems"] = self._past

        input_models = {}
        for key in ["input_ids", "labels"]:
            input_models[key] = inputs[key]

        return input_models

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps, scaler=scaler
            )
            return loss_mb.reduce_mean().detach().to(self.args.device)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

    def compute_loss(self, model, inputs, return_outputs=False, step=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        labels = inputs.pop("labels")
        # BCELoss必须用Float，具体是fp32，fp16还是bf16就看训练时用了什么

        # outputs[1]: pooled output
        outputs = model(**inputs)
        try: pooled_output = outputs[1]
        except: pooled_output = outputs["last_hidden_state"][:,1,:]
        pooled_output = self.dropout(pooled_output)

        # 只有Linear Sigmoid
        if self.model_args.model_type == "LinearSigmoid":
            logits = self.classifier(pooled_output)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            outputs = self.sigmoid(logits)
        # Linear Sigmoid，每次只训练正样本
        elif self.model_args.model_type == "LinearSigmoid_MultiLabel":
            logits = self.classifier(pooled_output)
            loss_fct = multilabel_categorical_crossentropy
            loss = loss_fct(logits, labels)
            outputs = self.sigmoid(logits)
        elif self.model_args.model_type == "LinearSigmoid_SparseMultiLabel":
            logits = self.classifier(pooled_output)
            loss_fct = sparse_multilabel_categorical_crossentropy
            loss = loss_fct(logits, labels)
            outputs = self.sigmoid(logits)
        # Linear Sigmoid，每次只训练正样本
        elif self.model_args.model_type == "LinearSigmoid_SoftLabel":
            logits = self.classifier(pooled_output)
            labels_list = labels.cpu().tolist()
            labels_index_list = []  # [0, ..., num_intent_labels]
            for row in range(len(labels_list)):
                labels_index_list.append(labels_list[row].index(1))

            logits_list = logits.cpu().tolist()
            for row in range(len(logits_list)):
                logits_list[row][labels_index_list[row]] = 1

            labels_fix = torch.Tensor(logits_list).to(self.model.device)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels_fix)
            outputs = self.sigmoid(logits)
        # Linear Sigmoid，采样1个负样本，每次训练1个正样本和1个负样本
        elif (
            self.model_args.model_type == "LinearSigmoid_SoftLabel_NegativeSample"
        ):
            logits = self.classifier(pooled_output) # [batch_size, label_size]
            outputs = self.sigmoid(logits) # [batch_size, label_size]
            labels_list = labels.cpu().tolist() # [batch_size, label_size]
            labels_index_list = []  # [0, ..., num_intent_labels]
            for row in range(len(labels_list)):
                labels_index_list.append(labels_list[row].index(1))

            logits_list = outputs.cpu().tolist()
            for row in range(len(logits_list)):
                logits_list[row][labels_index_list[row]] = 1

                negative_number_list = random_int_list(0, len(self.intent_label_set) - 1, self.model_args.negative_sample_num)

                for i in range(0, self.model_args.negative_sample_num, 1):
                    negative_number = negative_number_list[i]
                    if negative_number == labels_index_list[row] and negative_number != 0:
                        negative_number -= 1
                    elif (
                        negative_number == labels_index_list[row] and negative_number != len(self.intent_label_set) - 1
                    ):
                        negative_number += 1
                    logits_list[row][negative_number] = 0

            labels_fix = torch.Tensor(logits_list).to(self.model.device)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels_fix)
        elif (
            self.model_args.model_type == "LinearSigmoid_SoftLabel_TwoStep_NegativeSample"
        ):
            logits = self.classifier(pooled_output) # [batch_size, label_size]
            outputs = self.sigmoid(logits) # [batch_size, label_size]
            labels_list = labels.cpu().tolist() # [batch_size, label_size]
            labels_index_list = []  # [0, ..., num_intent_labels]
            for row in range(len(labels_list)):
                labels_index_list.append(labels_list[row].index(1))

            logits_list = outputs.cpu().tolist()
            if self.state.epoch < self.state.num_train_epochs * self.model_args.two_step_ratio:
                for row in range(len(logits_list)):
                    logits_list[row][labels_index_list[row]] = 1

                    negative_number_list = random_int_list(0, len(self.intent_label_set) - 1, self.model_args.negative_sample_num)

                    for i in range(0, self.model_args.negative_sample_num, 1):
                        negative_number = negative_number_list[i]
                        if negative_number == labels_index_list[row] and negative_number != 0:
                            negative_number -= 1
                        elif (
                            negative_number == labels_index_list[row] and negative_number != len(self.intent_label_set) - 1
                        ):
                            negative_number += 1
                        logits_list[row][negative_number] = 0
            else:
                for row in range(len(logits_list)):
                    logits_list[row][labels_index_list[row]] = 1

                    negative_number_list = minimum_int_list(logits_list[row], self.model_args.negative_sample_num)

                    for i in range(0, self.model_args.negative_sample_num, 1):
                        negative_number = negative_number_list[i]
                        if negative_number == labels_index_list[row] and negative_number != 0:
                            negative_number -= 1
                        elif (
                            negative_number == labels_index_list[row] and negative_number != len(self.intent_label_set) - 1
                        ):
                            negative_number += 1
                        logits_list[row][negative_number] = 0

            labels_fix = torch.Tensor(logits_list).to(self.model.device)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels_fix)
        elif self.model_args.model_type == "LinearSigmoid_FocalLoss":

            logits = self.classifier(pooled_output) # [batch_size, label_size]

            alpha = torch.Tensor([self.model_args.focal_loss_alpha_positive, self.model_args.focal_loss_alpha_negative]).to(self.model.device)
            gamma = self.model_args.focal_loss_gamma
            loss_fct = focal_loss(self.model_args.focal_loss_base_loss, self.model_args.focal_loss_label_smothing_rate, alpha, gamma)

            outputs = self.sigmoid(logits) # [batch_size, label_size]
            outputs_for_loss = torch.cat((torch.ones(outputs.shape).to(self.model.device).sub(outputs).unsqueeze(-1), outputs.unsqueeze(-1)), -1)

            loss = loss_fct(outputs_for_loss.reshape(-1, 2), labels.reshape(-1).long())

        outputs = {"loss": loss, "outputs": outputs}

        return (loss, outputs) if return_outputs else loss
