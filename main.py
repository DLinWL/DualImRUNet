import os
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from utils import FakeLR, Trainer_up, Tester_up, WarmUpCosineAnnealingLR
from utils import build_run_tag, init_device, init_model_up, logger, process_data
from utils.parser import args

# DATA_ROOT = Path("../DualImRUNet/dataset")
DATA_ROOT = Path("../DualImRUNet_dataset")


def _select_dataset_files():
    """Resolve dataset file names based on current CLI flags."""
    if args.eig_flag != 1:
        raise ValueError("Only eigenvector_flag=1 is supported with current dataset naming.")

    env_num = args.env_num
    suffix = "eigenvector"

    if args.ad_flag == 1 and args.enhanced_eigenvector_flag == 1:
        suffix += "_ad_enhanced"
    elif args.ad_flag == 1:
        suffix += "_ad"
    elif args.enhanced_eigenvector_flag == 1:
        # No matching test files provided for enhanced-only; block to avoid silent failure.
        raise ValueError("enhanced_eigenvector_flag=1 without ad_flag=1 is not supported by available datasets.")

    trainval_filename = DATA_ROOT / f"trainvalset_env{env_num}_{suffix}.npy"
    test_filename = DATA_ROOT / f"testset_env30_{suffix}.npy"
    trainval_filename_up = DATA_ROOT / f"trainvalset_env{env_num}_{suffix}_uplink.npy"
    test_filename_up = DATA_ROOT / f"testset_env30_{suffix}_uplink.npy"
    return (
        str(trainval_filename),
        str(test_filename),
        str(trainval_filename_up),
        str(test_filename_up),
    )


def _build_dataloaders(train_data, val_data, test_data, train_data_up, val_data_up, test_data_up, batch_size, pin_memory):
    def to_mag(data):
        return abs(data[:, 0, :, :] + 1j * (data[:, 1, :, :]))

    train_data_up_mag = to_mag(train_data_up)
    val_data_up_mag = to_mag(val_data_up)
    test_data_up_mag = to_mag(test_data_up)

    train_dataset_tensor = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_data_up_mag, dtype=torch.float32),
    )
    val_dataset_tensor = TensorDataset(
        torch.tensor(val_data, dtype=torch.float32),
        torch.tensor(val_data_up_mag, dtype=torch.float32),
    )
    test_dataset_tensor = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_data_up_mag, dtype=torch.float32),
        torch.tensor(test_data, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset_tensor, batch_size=batch_size, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset_tensor, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)
    test_loader = DataLoader(test_dataset_tensor, batch_size=batch_size, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader, test_loader


def main():
    logger.info(f"=> PyTorch Version: {torch.__version__}")

    device, pin_memory = init_device(args.seed, args.cpu, args.gpu, args.cpu_affinity)
    batch_size = 200

    (
        trainval_filename,
        test_filename,
        trainval_filename_up,
        test_filename_up,
    ) = _select_dataset_files()

    train_data, val_data, test_data = process_data(
        trainval_filename, test_filename, args.spalign_flag
    )
    train_data_up, val_data_up, test_data_up = process_data(
        trainval_filename_up, test_filename_up, args.spalign_flag
    )
    train_loader, val_loader, test_loader = _build_dataloaders(
        train_data, val_data, test_data, train_data_up, val_data_up, test_data_up, batch_size, pin_memory
    )

    model = init_model_up(args)
    model.to(device)

    criterion = nn.MSELoss().to(device)
    lr_init = 1e-4 if args.scheduler == 'const' else 2e-4
    optimizer = torch.optim.Adam(model.parameters(), lr_init)

    if args.scheduler == 'const':
        scheduler = FakeLR(optimizer=optimizer)
    else:
        scheduler = WarmUpCosineAnnealingLR(
            optimizer=optimizer,
            T_max=args.epochs * len(train_loader),
            T_warmup=60 * len(train_loader),
            eta_min=5e-5,
        )

    save_path = str(Path("./checkpoints") / build_run_tag(args))
    trainer = Trainer_up(
        model=model,
        device=device,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        save_path=save_path,
        resume=args.resume,
    )

    trainer.loop(args.epochs, train_loader, val_loader, test_loader)

    loss, rho, nmse = Tester_up(model, device, criterion)(test_loader)
    print(
        f"\n=! Final test loss: {loss:.3e}"
        f"\n         test rho: {rho:.3e}"
        f"\n         test NMSE: {nmse:.3e}\n"
    )


if __name__ == "__main__":
    main()
