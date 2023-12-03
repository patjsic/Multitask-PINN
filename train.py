import os
import torch
import argparse
import numpy as np

from data import get_navier_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from models.pinn import PINN
from models.multitask_pinn import MTLPINN

def train(epochs=30, hidden_size=32, batch_size=1024, lr=1e-3, device="auto", mtl=False, noise=0.1):
    #Check for device if set to auto
    if device=="auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #Load tensorboard writer and create runs directory
    writer = SummaryWriter()

    #Load data, error handling in `load_data()`
    print("Loading data...")
    train_dataset, test_dataset, __, __ = get_navier_data("data", noise)

    #Create dataloader objects
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    #Training Loop
    print("Training model...")
    if mtl:
        model = MTLPINN(3, hidden_size, 2, device=device)
        print(f"Training MTL PINN with num parameters: {sum(p.numel() for p in model.parameters())}")
    else:
        model = PINN(3, hidden_size, 2, device=device)
        print(f"Training Vanilla PINN with num parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    losses = []
    for epoch in tqdm(range(epochs)):
        optimizer.zero_grad()

        for idx, s in enumerate(train_dataloader):
            #Get components
            x = s["x"]
            y = s["y"]
            t = s["t"]
            p = s["p"]
            u = s["u"]
            v = s["v"]

            #Make forward pass and backprop
            pred = model.forward(x, y, t, p, u, v)
            loss = model.compute_loss(u, v, pred["u_pred"], pred["v_pred"], pred["f"], pred["g"])
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar("Pde_loss/train", model.pde_loss_val, epoch)
            writer.add_scalar("Data_loss/train", model.data_loss_val, epoch)

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, loss {loss}")
            
    writer.flush()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", dest="epochs", type=int, default=30, help="number of training iterations, default is 30")
    parser.add_argument("--hidden", dest="hidden", type=int, default=32, help="size of hidden layers, default is 32")
    parser.add_argument("--batch", dest="batch", type=int, default=1024, help="number of examples per batch, default is 1024")
    parser.add_argument("--lr", dest="lr", type=float, default=1e-3, help="learning rate, default is 0.001")
    parser.add_argument("--device", dest="device", type=str, default="auto", choice=["auto", "cpu", "cuda"], help="training device")
    parser.add_argument("--mtl", dest="mtl", type=bool, default=True, action="store_true", help="if true, train MTL model else train vanilla PINN")
    parser.add_argument("--noise", dest="noise", type=float, default=0.1, help="amount of noise in data for u and v")
    args = parser.parse_args()
    train(epochs=args.epochs, hidden_size=args.hidden, batch_size=args.batch, lr=args.lr, device=args.device, mtl=args.mtl, noise=args.noise)