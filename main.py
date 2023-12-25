import wandb
import torch
from tqdm import tqdm

from models import TMeanNet
from datasets import get_dvlog_dataloader


def train_epoch(
    net, train_loader, loss_fn, optimizer, scheduler, device, 
    current_epoch, total_epochs
):
    """One training epoch.
    """
    net.train()
    sample_count = 0
    running_loss = 0.
    correct_count = 0

    with tqdm(
        train_loader, desc=f"Training epoch {current_epoch}/{total_epochs}",
        leave=False, unit="batch"
    ) as pbar:
        for x, y in pbar:
            x, y = x.to(device), y.to(device).unsqueeze(1)
            y_pred = net(x)
            loss = loss_fn(y_pred, y.to(torch.float32))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            sample_count += x.shape[0]
            running_loss += loss.item() * x.shape[0]
            # binary classification with only one output neuron
            pred = (y_pred > 0).int()
            correct_count += (pred == y).sum().item()

            pbar.set_postfix({
                "loss": running_loss / sample_count,
                "acc": correct_count / sample_count,
            })

    if scheduler is not None:
        scheduler.step()

    return {
        "loss": running_loss / sample_count,
        "acc": correct_count / sample_count,
    }


def val(
    net, val_loader, loss_fn, device, 
):
    """Test the model on the validation / test set.
    """
    net.eval()
    sample_count = 0
    running_loss = 0.
    TP, FP, TN, FN = 0, 0, 0, 0

    with torch.no_grad():
        with tqdm(
            val_loader, desc="Validating", leave=False, unit="batch"
        ) as pbar:
            for x, y in pbar:
                x, y = x.to(device), y.to(device).unsqueeze(1)
                y_pred = net(x)
                loss = loss_fn(y_pred, y.to(torch.float32))

                sample_count += x.shape[0]
                running_loss += loss.item() * x.shape[0]
                # binary classification with only one output neuron
                pred = (y_pred > 0).int()
                TP += torch.sum((pred == 1) & (y == 1)).item()
                FP += torch.sum((pred == 1) & (y == 0)).item()
                TN += torch.sum((pred == 0) & (y == 0)).item()
                FN += torch.sum((pred == 0) & (y == 1)).item()

                l = running_loss / sample_count
                precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
                recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
                f1_score = (
                    2 * (precision * recall) / (precision + recall) 
                    if (precision + recall) > 0 else 0.0
                )
                accuracy = (
                    (TP + TN) / sample_count
                    if sample_count > 0 else 0.0
                )

                pbar.set_postfix({
                    "loss": l, "acc": accuracy,
                    "precision": precision, "recall": recall, "f1": f1_score,
                })

    l = running_loss / sample_count
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = (
        2 * (precision * recall) / (precision + recall) 
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (
        (TP + TN) / sample_count
        if sample_count > 0 else 0.0
    )
    return {
        "loss": l, "acc": accuracy,
        "precision": precision, "recall": recall, "f1": f1_score,
    }


def main():
    net = TMeanNet(hidden_sizes=[512, 512, 512])
    train_loader = get_dvlog_dataloader("../dvlog-dataset", "train")
    val_loader = get_dvlog_dataloader("../dvlog-dataset", "valid")
    test_loader = get_dvlog_dataloader("../dvlog-dataset", "test")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

    for epoch in range(500):
        train_results = train_epoch(
            net, train_loader, loss_fn, optimizer, None, "cpu", epoch, 100
        )
        val_results = val(net, val_loader, loss_fn, "cpu")
        print(f"Epoch {epoch} results:")
        print(train_results)
        print(val_results)

    test_results = val(net, test_loader, loss_fn, "cpu")
    print("Test results:")
    print(test_results)



if __name__ == '__main__':
    main()