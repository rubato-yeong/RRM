"""
    2_train_TopKSAE.py

    Train Top-K Sparse Autoencoders (TopKSAE) on pre-extracted transformer activations from ViT-like models (e.g., ViT, DINOv2, CLIP-ViT)

    Main functionalities:
        - Supports multiple combinations of top-K sparsity and dictionary size (R)
        - Loads pre-saved activation datasets grouped by file (e.g., from ImageNet)
        - Performs L1-sparse autoencoder training with optional auxiliary top-k penalty
        - Logs training metrics and saves model checkpoints periodically
        - Supports optional validation split
        - Supports checkpoint loading and resume from arbitrary epoch

    Arguments:
        --top_k: List of top-k values (sparsity constraint)
        --dict_size_R: List of dictionary sizes (number of latent features)
        --acts_data_path: Root path to pre-extracted activations organized by [train|valid]/[model]/layer{L}  (e.g., /USER/activations/imagenet/)
        --model_layer: Layer index from which activations were extracted
        --model_type: One of ['vit', 'dinov2', 'clip_vit']
        --save_dir: Path to directory where model checkpoints are saved  (e.g., /USER/checkpoints/topk_sae/)
        --num_epochs: Number of training epochs
        --log_every: Print/log loss every N batches
        --save_every: Save checkpoint every N epochs
        --use_wandb: Whether to log metrics to Weights & Biases
        --epoch_start_from: Used to resume training from a specific epoch
        --do_validation: Enable evaluation on the validation set

    Output:
        - Saves checkpoints as .pth files for each (top_k, dict_size_R) combination
        - Logs metrics including loss, L1 penalty, L2 reconstruction loss, dead unit counts
        - Optionally reports validation loss and logs to wandb

    Example:
        python 2_train_topk_sae.py --model_type vit --model_layer 10 --top_k 64 128 --dict_size_R 1024 2048 --acts_data_path {/path/to/activations} --do_validation
"""


import argparse
import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm

from utils import TopKSAE, set_seed, GroupedActivationDataset


YOUR_WANDB_API_KEY = ...
YOUR_WANDB_ENTITY_NAME = ...
YOUR_WANDB_PROJECT_NAME = ...


def parse_args():
    parser = argparse.ArgumentParser(description="Train Top-K Sparse Autoencoder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--act_size", type=int, default=768, help="Input feature size (e.g., ViT activation size).")
    parser.add_argument("--dict_size_R", type=float, nargs='+', required=True, help="List of dictionary sizes (latent dimensions).")
    parser.add_argument("--top_k", type=int, nargs='+', required=True, help="List of top_k values.")
    parser.add_argument("--top_k_aux", type=int, default=256, help="Number of auxiliary top-k activations.")
    parser.add_argument("--n_batches_to_dead", type=int, default=60, help="Number of batches before a feature is considered dead.")
    parser.add_argument("--aux_penalty", type=float, default=1/32, help="Auxiliary loss coefficient.")
    parser.add_argument("--l1_coeff", type=float, default=1e-3, help="L1 regularization coefficient.")
    parser.add_argument("--input_unit_norm", default=True, type=bool, help="Normalize input features.")
    parser.add_argument("--activation_group_size", type=int, default=1, help="Number of files per activation group.")
    parser.add_argument("--model_layer", type=int, required=True, help="Layer number for activation data path.")
    parser.add_argument("--num_epochs", type=int, default=400, help="Number of training epochs.")
    parser.add_argument("--log_every", type=int, default=10, help="Logging frequency (in batches).")
    parser.add_argument("--save_every", type=int, default=5, help="Checkpoint saving frequency (in epochs).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for optimizer.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train the sparse autoencoder on.")
    parser.add_argument("--use_wandb", type=bool, default=True, help="Whether to use Weights & Biases for logging.")
    parser.add_argument("--epoch_start_from", type=int, default=1, help="Which epoch to start from (for resuming training).")
    parser.add_argument("--do_validation", action="store_true", help="Whether to perform validation.")
    parser.add_argument("--model_type", type=str, required=True, choices=["vit", "dinov2", "clip_vit"], help="Model architecture.")

    parser.add_argument("--save_dir", type=str, default="", help="Directory to save checkpoints.")  # E.g., PROJECT_ROOT/checkpoints/
    parser.add_argument("--acts_data_path", type=str, default="", help="Activation data path. Use a path that has created with 1_extract_activations.py.")  # E.g., PROJECT_ROOT/activations/imagenet/
    
    return vars(parser.parse_args())


def train(args, models, dataloader_train, dataloader_val, optimizers, save_paths, model_ids, dataset_name):
    if args["use_wandb"]:
        wandb.init(
            entity=YOUR_WANDB_ENTITY_NAME,
            project=YOUR_WANDB_PROJECT_NAME,
            name=f"{args['model_type']}_{dataset_name}_K{args['top_k'][0]}-{args['top_k'][-1]}_R{args['dict_size_R'][0]}-{args['dict_size_R'][-1]}_L{args['model_layer']}",
            config=args,
            resume="never",
        )

    num_epochs = args["num_epochs"]
    num_batches = len(dataloader_train)

    for model in models:
        model.train()

    for epoch in range(1, num_epochs + 1):
        if epoch < args["epoch_start_from"]:
            continue

        epoch_losses = {model_id: 0.0 for model_id in model_ids}
        total_sample_size = 0

        for i, batch in enumerate(dataloader_train):
            batch = batch.to(args["device"])

            for model, optimizer, model_id in zip(models, optimizers, model_ids):
                optimizer.zero_grad()
                output = model(batch)
                loss = output["loss"]
                loss.backward()
                optimizer.step()
                
                model.make_decoder_weights_and_grad_unit_norm()
                epoch_losses[model_id] += loss.item() * batch.size(0)

                if i % args["log_every"] == 0:
                    print(f"Epoch {epoch}/{num_epochs} | Batch {i}/{num_batches} | {model_id} Loss: {loss.item():.4f}")
                    if args["use_wandb"]:
                        wandb.log({
                            "epoch": epoch,
                            f"{model_id}_loss": loss.item(),
                            f"{model_id}_l1_loss": output["l1_loss"].item(),
                            f"{model_id}_l2_loss": output["l2_loss"].item(),
                            f"{model_id}_l0_norm": output["l0_norm"].item(),
                            f"{model_id}_num_dead_features": output["num_dead_features"].item(),
                        })

            total_sample_size += batch.size(0)

        for model_id in model_ids:
            avg_loss = epoch_losses[model_id] / total_sample_size
            print(f"Epoch {epoch} Average Loss for {model_id}: {avg_loss:.4f}")
            if args["use_wandb"]:
                wandb.log({
                    "epoch": epoch,
                    f"{model_id}_avg_loss": avg_loss,
                })

        # Validation
        if args["do_validation"] and dataloader_val is not None:
            for model, model_id in zip(models, model_ids):
                avg_val_loss = validate(args, model, dataloader_val)
                print(f"Epoch {epoch} Validation Loss for {model_id}: {avg_val_loss:.4f}")
                if args["use_wandb"]:
                    wandb.log({
                        "epoch": epoch,
                        f"{model_id}_val_loss": avg_val_loss,
                    })

        # Save checkpoints
        for model, optimizer, save_path, model_id in zip(models, optimizers, save_paths, model_ids):
            if epoch % args["save_every"] == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                checkpoint_path = os.path.join(save_path, f"epoch{epoch}.pth")
                torch.save(checkpoint, checkpoint_path)
                print(f"Checkpoint for {model_id} saved at epoch {epoch} to {checkpoint_path}")

    if args["use_wandb"]:
        wandb.finish()
    
def validate(cfg, model, dataloader):
    model.eval()
    epoch_loss = 0.0
    total_sample_num = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating..."):
            batch = batch.to(cfg["device"])
            output = model(batch)
            loss = output["l2_loss"]
            
            batch_len = batch.shape[0]
            epoch_loss += loss.item() * batch_len
            total_sample_num += batch_len
        
    avg_loss = epoch_loss / total_sample_num
    return avg_loss

def main():
    args = parse_args()
    set_seed(args["seed"])

    if args["use_wandb"]:
        wandb.login(key=YOUR_WANDB_API_KEY)
    
    device = torch.device(args["device"] if torch.cuda.is_available() or "cpu" in args["device"] else "cpu")
    args["device"] = device
    args["dtype"] = torch.float32
    
    if "imagenet" in args["acts_data_path"].lower():
        dataset_name = "imagenet"
    else:
        raise NotImplementedError(f"Unsupported dataset name in {args['acts_data_path']}")
    
    dataset_train_path = os.path.join(args["acts_data_path"], f"train/{args['model_type']}/layer{args['model_layer']}")
    activation_dataset_train = GroupedActivationDataset(dataset_train_path, group_size=args["activation_group_size"])
    dataloader_train = DataLoader(
        activation_dataset_train,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4
    )
    
    if args["do_validation"]:
        dataset_val_path = os.path.join(args["acts_data_path"], f"valid/{args['model_type']}/layer{args['model_layer']}")
        activation_dataset_val = GroupedActivationDataset(dataset_val_path, group_size=args["activation_group_size"])
        dataloader_val = DataLoader(
            activation_dataset_val,
            batch_size=args["batch_size"],
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2
        )
    else:
        dataloader_val = None
        
        
    models = []
    optimizers = []
    model_ids = []
    save_paths = []
    
    for k in args["top_k"]:
        for R in args["dict_size_R"]:
            model_args = args.copy()
            model_args["top_k"] = k
            model_args["dict_size_R"] = R
            model = TopKSAE(model_args).to(device)
            models.append(model)
            
            optimizer = optim.Adam(model.parameters(), lr=args["learning_rate"], betas=(0.9, 0.999))
            optimizers.append(optimizer)
            model_ids.append(f"{model_args['model_type']}_K{k}_R{R}_L{args['model_layer']}")
            
            save_path = os.path.join(
                args["save_dir"],
                dataset_name,
                model_args["model_type"],
                f"K{k}_R{R}",
                f"L{args['model_layer']}"
            )
            save_paths.append(save_path)
            os.makedirs(save_path, exist_ok=True)
    
    if args["epoch_start_from"] > 1:
        for model, optimizer, save_path, model_id in zip(models, optimizers, save_paths, model_ids):
            checkpoint_file = os.path.join(
                save_path,
                f"epoch{args['epoch_start_from']}.pth"
            )
            print(f"Loading checkpoint for {model_id} from {checkpoint_file}")
            checkpoint = torch.load(checkpoint_file, map_location=args["device"])
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Training loop
    train(
        args,
        models, 
        dataloader_train,
        dataloader_val,
        optimizers,
        save_paths,
        model_ids,
        dataset_name,
    )


if __name__ == '__main__':
    main()
