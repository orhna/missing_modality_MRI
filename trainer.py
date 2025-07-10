import torch
from tqdm import tqdm
import itertools
import os
import json
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
import utils.utils as utils

class Trainer:
    """
    Trainer class to handle training and validation loops.
    """

    def __init__(
        self, model, optimizer, scheduler, loss_function, dice_metric_c, sensitivity_metric,
        precision_metric, IOU_metric, hausdorff_metric, dice_dict_m, post_trans,
        modalities_to_train, device, writer, training_config, model_save_path, chosen_ds,
        counter_dict, train_size, cropped_input_size
    ):
        """
        Initialize the Trainer.

        Args:
            model (torch.nn.Module): Model to train.
            optimizer (torch.optim.Optimizer): Optimizer.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            loss_function (callable): Loss function.
            dice_metric_c, sensitivity_metric, precision_metric, IOU_metric, hausdorff_metric: Metric objects.
            dice_dict_m (dict): Dictionary of DiceMetric objects for missing modalities.
            post_trans (callable): Post-processing transform.
            modalities_to_train (list): Modalities to use for training.
            device (torch.device): Device to run on.
            writer (SummaryWriter): Tensorboard writer.
            training_config (object): Training configuration.
            model_save_path (str): Path to save models.
            chosen_ds (str): Dataset identifier.
            counter_dict (dict): Dictionary to count dropped modalities.
            train_size (int): Number of training samples.
            cropped_input_size (tuple): Input size for cropping.
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_function = loss_function
        self.dice_metric_c = dice_metric_c
        self.sensitivity_metric = sensitivity_metric
        self.precision_metric = precision_metric
        self.IOU_metric = IOU_metric
        self.hausdorff_metric = hausdorff_metric
        self.dice_dict_m = dice_dict_m
        self.post_trans = post_trans
        self.modalities_to_train = modalities_to_train
        self.device = device
        self.writer = writer
        self.training_config = training_config
        self.model_save_path = model_save_path
        self.chosen_ds = chosen_ds
        self.counter_dict = counter_dict
        self.train_size = train_size
        self.cropped_input_size = cropped_input_size

    def train_one_epoch(self, train_loader, epoch):
        """
        Run a single training epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epoch (int): Current epoch number.

        Returns:
            tuple: (epoch_loss, epoch_ds_loss, epoch_sep_dec_loss)
        """
        self.model.train()
        self.model.is_training = True
        epoch_loss = 0
        epoch_ds_loss = 0
        epoch_sep_dec_loss = 0
        step = 0
        img_index = 0
        label_index = 1

        for batch in tqdm(train_loader):
            self.optimizer.zero_grad()
            step += 1
            label = batch[label_index].to(self.device)
            batch[img_index], d_m = utils.rand_drop_channel(
                self.modalities_to_train, batch[img_index], mode="modmean"
            )
            input_data = batch[img_index].to(self.device)
            self.counter_dict[d_m] += 1

            out, ds_outs, sep_out = self.model(input_data, modalities_dropped_info=d_m)

            loss = self.loss_function(out, label)
            ds_loss_0 = self.loss_function(ds_outs[0], label)
            ds_loss_1 = self.loss_function(ds_outs[1], label)
            ds_loss_2 = self.loss_function(ds_outs[2], label)
            ds_loss_3 = self.loss_function(ds_outs[3], label)
            ds_loss_4 = self.loss_function(ds_outs[4], label)

            f_loss = self.loss_function(sep_out[0], label)
            t1c_loss = self.loss_function(sep_out[1], label)
            t1_loss = self.loss_function(sep_out[2], label)
            t2_loss = self.loss_function(sep_out[3], label)
            sep_losses = f_loss + t1c_loss + t1_loss + t2_loss
            _w = 0.2
            ds_loss = _w * ds_loss_0 + _w * ds_loss_1 + _w * ds_loss_2 + _w * ds_loss_3 + _w * ds_loss_4
            total_loss = loss + ds_loss + sep_losses

            total_loss.backward()
            self.optimizer.step()

            epoch_loss += total_loss.item()
            epoch_len = self.train_size // self.training_config.train_batch_size
            epoch_ds_loss += ds_loss.item()
            epoch_sep_dec_loss += sep_losses.item()

            print(f"{step}/{epoch_len}, train_loss: {total_loss.item():.4f}")

        epoch_loss /= step
        epoch_ds_loss /= step
        epoch_sep_dec_loss /= step
        return epoch_loss, epoch_ds_loss, epoch_sep_dec_loss

    def validate(self, val_loader, epoch, best_metrics):
        """
        Run validation and update best metrics.

        Args:
            val_loader (DataLoader): DataLoader for validation data.
            epoch (int): Current epoch number.
            best_metrics (dict): Dictionary of best metrics.

        Returns:
            dict: Updated best_metrics.
        """
        self.model.eval()
        self.model.is_training = False
        with torch.no_grad():
            metric_c = {}
            metric_m = {}
            dict_to_save = {}

            for val_data in val_loader:
                val_input = val_data[0]
                complete_modality_image = val_input.to(self.device)
                val_labels = val_data[1].to(self.device)
                c_logits = sliding_window_inference(
                    complete_modality_image, roi_size=(128, 128, 128), sw_batch_size=1,
                    predictor=lambda x: self.model(x, modalities_dropped_info=[])
                )
                c_outputs = [self.post_trans(i) for i in decollate_batch(c_logits)]
                self.dice_metric_c(y_pred=c_outputs, y=val_labels)
                self.sensitivity_metric(y_pred=c_outputs, y=val_labels)
                self.precision_metric(y_pred=c_outputs, y=val_labels)
                self.IOU_metric(y_pred=c_outputs, y=val_labels)
                self.hausdorff_metric(y_pred=c_outputs, y=val_labels)

                for r in range(1, 4):
                    for channels_to_drop_tuple in itertools.combinations(range(4), r):
                        modalities_to_drop_list = list(channels_to_drop_tuple)
                        remaining_channels = [x for x in range(4) if x not in modalities_to_drop_list]
                        missing_modality_image = utils.drop_modality_image_channel(
                            complete_modality_image, method="modality_mean",
                            idx_to_drop=modalities_to_drop_list, remaining_modalities=remaining_channels
                        )
                        m_logits = sliding_window_inference(
                            missing_modality_image, roi_size=(128, 128, 128), sw_batch_size=1,
                            predictor=lambda x: self.model(x, modalities_dropped_info=modalities_to_drop_list)
                        )
                        m_outputs = [self.post_trans(i) for i in decollate_batch(m_logits)]
                        _key = utils.get_dict_key(remaining_channels)
                        self.dice_dict_m["".join(_key)](y_pred=m_outputs, y=val_labels)

            for sc_key in self.dice_dict_m:
                metric_m["dice_" + sc_key] = self.dice_dict_m[sc_key].aggregate()
                self.writer.add_scalar(f"Val/{sc_key}/Dice_M_ET", metric_m["dice_" + sc_key][2].item(), epoch)
                self.writer.add_scalar(f"Val/{sc_key}/Dice_M_TC", metric_m["dice_" + sc_key][0].item(), epoch)
                self.writer.add_scalar(f"Val/{sc_key}/Dice_M_WT", metric_m["dice_" + sc_key][1].item(), epoch)
                dict_to_save[sc_key] = {
                    "Dice_M_ET": metric_m["dice_" + sc_key][2].item(),
                    "Dice_M_TC": metric_m["dice_" + sc_key][0].item(),
                    "Dice_M_WT": metric_m["dice_" + sc_key][1].item()
                }
            for sc_key in self.dice_dict_m:
                self.dice_dict_m[sc_key].reset()

            metric_c["dice"] = self.dice_metric_c.aggregate()
            metric_c["hausdorff_distance"] = self.hausdorff_metric.aggregate()
            metric_c["sensitivity"] = self.sensitivity_metric.aggregate()[0].item()
            metric_c["precision"] = self.precision_metric.aggregate()[0].item()
            metric_c["IOU"] = self.IOU_metric.aggregate().item()
            metric_c["dice_ET"] = self.dice_metric_c.aggregate()[2].item()
            metric_c["dice_TC"] = self.dice_metric_c.aggregate()[0].item()
            metric_c["dice_WT"] = self.dice_metric_c.aggregate()[1].item()
            metric_c["HD_ET"] = self.hausdorff_metric.aggregate()[2].item()
            metric_c["HD_TC"] = self.hausdorff_metric.aggregate()[0].item()
            metric_c["HD_WT"] = self.hausdorff_metric.aggregate()[1].item()
            utils.log_visuals(self.modalities_to_train, complete_modality_image, c_outputs[0], val_labels, epoch, self.writer, self.chosen_ds)
            utils.log_metrics(self.writer, metric_c, epoch, self.chosen_ds)

            dict_to_save["complete"] = {
                "Dice_M_ET": metric_c["dice_ET"],
                "Dice_M_TC": metric_c["dice_TC"],
                "Dice_M_WT": metric_c["dice_WT"]
            }
            values = [inner_dict["Dice_M_ET"] for inner_dict in dict_to_save.values()]
            avg_ET = sum(values) / len(values)
            values = [inner_dict["Dice_M_TC"] for inner_dict in dict_to_save.values()]
            avg_TC = sum(values) / len(values)
            values = [inner_dict["Dice_M_WT"] for inner_dict in dict_to_save.values()]
            avg_WT = sum(values) / len(values)
            self.writer.add_scalar(f"Val/Avg/Dice_M_ET", avg_ET, epoch)
            self.writer.add_scalar(f"Val/Avg/Dice_M_TC", avg_TC, epoch)
            self.writer.add_scalar(f"Val/Avg/Dice_M_WT", avg_WT, epoch)
            dict_to_save["average"] = {"Dice_ET": avg_ET, "Dice_TC": avg_TC, "Dice_WT": avg_WT}

            self.dice_metric_c.reset()
            self.hausdorff_metric.reset()
            self.sensitivity_metric.reset()
            self.precision_metric.reset()
            self.IOU_metric.reset()

            json_dump_path = os.path.dirname(self.model_save_path) + f"/{str(epoch)}_dice_results.json"
            with open(json_dump_path, "w") as json_file:
                json.dump(dict_to_save, json_file, indent=4)

            # Save best models
            if avg_TC > best_metrics["TC"]:
                best_metrics["TC"] = avg_TC
                if epoch > 1:
                    model_save_name = self.model_save_path + self.training_config.experiment_name + "_BEST_TC.pth"
                    torch.save(self.model.state_dict(), model_save_name)
            if avg_WT > best_metrics["WT"]:
                best_metrics["WT"] = avg_WT
                if epoch > 1:
                    model_save_name = self.model_save_path + self.training_config.experiment_name + "_BEST_WT.pth"
                    torch.save(self.model.state_dict(), model_save_name)
            if avg_ET > best_metrics["ET"]:
                best_metrics["ET"] = avg_ET
                if epoch > 1:
                    model_save_name = self.model_save_path + self.training_config.experiment_name + "_BEST_ET.pth"
                    torch.save(self.model.state_dict(), model_save_name)
        return best_metrics