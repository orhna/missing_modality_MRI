import torch
from glob import glob
import os
from tqdm import tqdm
from datetime import datetime
import utils.utils as utils
import config
import argparse
import itertools
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from nets.multimodal_swinunetr_multirecon import Multimodal_SwinUNETR
from torch.optim.lr_scheduler import StepLR
from monai.losses.dice import DiceCELoss
from tensorboardX import SummaryWriter

from trainer import Trainer

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="ID of the GPU", type=str)
    return parser.parse_args()

def load_config():
    """
    Load training and database configuration.
    """
    training_config = config.Training_config()
    Database_config = config.Database_config()
    return training_config, Database_config

def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = parse_args()
    device_list = list(map(int, args.device_id.split(',')))

    # Load config
    training_config, Database_config = load_config()
    _date = datetime.now().strftime("%d-%H-%M")
    log_dir = f"./logs/joint/{training_config.experiment_name}_{_date}"
    writer = SummaryWriter(log_dir=log_dir)

    # Save config and set seed
    utils.save_config_from_py("config.py", log_dir)
    utils.set_random_seed(training_config.seed)

    cropped_input_size = training_config.cropped_input_size
    epochs = training_config.epoch

    # Set the data size and total modalities
    chosen_ds = training_config.dataset_to_train[0]

    # Path initialization
    img_path = Database_config.img_path[chosen_ds]
    seg_path = Database_config.seg_path[chosen_ds]
    model_save_path = training_config.model_save_path + training_config.experiment_name + "/"
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    images = sorted(glob(os.path.join(img_path, "*.*")))
    segs = sorted(glob(os.path.join(seg_path, "*.*")))

    # Create dict to count dropped modalities frequency
    counter_dict = utils.create_counter_dict()

    channel_indices = []
    modalities_to_train = training_config.modalities_to_train
    for _m in modalities_to_train:
        channel_indices.append(Database_config.channels[chosen_ds].index(_m))

    train_loader, val_loader = utils.get_loaders(
        images=images,
        segs=segs,
        train_file=Database_config.split_path[chosen_ds]["train"],
        val_file=Database_config.split_path[chosen_ds]["val"],
        workers=training_config.workers,
        train_batch_size=training_config.train_batch_size,
        cropped_input_size=cropped_input_size,
        channel_indices=channel_indices
    )

    train_size = len(train_loader.dataset)

    # Initialize GPU
    device = utils.initialize_GPU(device_list[0])

    # Initialize loss and metrics
    loss_function, dice_metric_c, sensitivity_metric, precision_metric, IOU_metric, post_trans = utils.initialize_loss_metric()
    hausdorff_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch", get_not_nans=False, percentile=95)
    loss_function = DiceCELoss(sigmoid=True, to_onehot_y=False)

    binary_combinations = [''.join(map(str, bits)) for bits in itertools.product([0, 1], repeat=4)]
    binary_combinations = [comb for comb in binary_combinations if comb != '0000' and comb != '1111']
    dice_dict_m = {comb: DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False) for comb in binary_combinations}

    epoched = 0
    model = Multimodal_SwinUNETR(
        img_size=(cropped_input_size[0], cropped_input_size[1], cropped_input_size[2]),
        in_channels=1,
        out_channels=training_config.output_channel,
        feature_size=training_config.feature_size,
        use_checkpoint=True,
        recon_level=training_config.recon_level,
        device=device
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.lr, weight_decay=training_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=training_config.sch_step_size, gamma=training_config.sch_gamma)

    if training_config.continue_training:
        print("Continuing training from checkpoint")
        checkpoint = torch.load(training_config.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoched = checkpoint['epoch'] + 1

    # Initialize the best metric
    best_metrics = {"ET": -1, "WT": -1, "TC": -1}

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        dice_metric_c=dice_metric_c,
        sensitivity_metric=sensitivity_metric,
        precision_metric=precision_metric,
        IOU_metric=IOU_metric,
        hausdorff_metric=hausdorff_metric,
        dice_dict_m=dice_dict_m,
        post_trans=post_trans,
        modalities_to_train=modalities_to_train,
        device=device,
        writer=writer,
        training_config=training_config,
        model_save_path=model_save_path,
        chosen_ds=chosen_ds,
        counter_dict=counter_dict,
        train_size=train_size,
        cropped_input_size=cropped_input_size
    )

    # Training loop
    for epoch in tqdm(range(epoched, epochs)):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epochs}")

        epoch_loss, epoch_ds_loss, epoch_sep_dec_loss = trainer.train_one_epoch(train_loader, epoch)

        if training_config.lr_scheduler:
            scheduler.step()
            writer.add_scalar("Training/EpochLR", optimizer.param_groups[0]['lr'], epoch)

        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        writer.add_scalar("Training/EpochLoss", epoch_loss, epoch)
        writer.add_scalar("Training/EpochLossDS", epoch_ds_loss, epoch)
        writer.add_scalar("Training/EpochLossSD", epoch_sep_dec_loss, epoch)

        # Save model
        if not epoch == 0 and epoch % 5 == 0:
            model_save_name = model_save_path + training_config.experiment_name + "_Epoch_" + str(epoch) + ".pth"
            opt_save_name = model_save_path + training_config.experiment_name + "_checkpoint_Epoch_" + str(epoch) + ".pt"
            torch.save(model.state_dict(), model_save_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
            }, opt_save_name)
            print("Saved Model")

        # Validation
        if epoch % training_config.val_interval == 0:
            best_metrics = trainer.validate(val_loader, epoch, best_metrics)

    writer.close()
    utils.export_counter_dict(counter_dict, log_dir)
    print(training_config.experiment_name)

if __name__ == "__main__":
    main()