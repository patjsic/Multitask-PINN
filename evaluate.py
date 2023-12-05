import os
import torch
import argparse
import numpy as np

def calculate_error(save_dir, test_data, preds, lambda1, lambda2):
    test_arr = np.array(test_data.data)
    p = test_arr[:, 3]
    u = test_arr[:, 4]
    v = test_arr[:, 5]
    p_pred = preds["p_pred"]
    u_pred = preds["u_pred"]
    v_pred = preds["v_pred"]

    log = ""

    #Error calculation
    u_error = np.mean(np.abs((u - u_pred) / u))
    v_error = np.mean(np.abs((v - v_pred) / v))
    p_error = np.mean(np.abs((p - p_pred) / p))
    lam1_error = np.abs(lambda1 - 1.0)
    lam2_error = np.abs(lambda2 - 0.01) / 0.01

    log += f"Error in u: {u_error} \n"
    log += f"Error in v: {v_error} \n"
    log += f"Error in p: {p_error} \n"
    log += f"Error in lambda_1: {lam1_error} \n"
    log += f"Error in lambda_2: {lam2_error} \n"

    with open(os.path.join(save_dir, "error_log.txt"), "w") as file:
        file.write(log)
    
if __name__ == "__main__":

    from train import train
    from data import *
    from torch.utils.data import DataLoader

    _, test_data, _, _ = get_navier_data("data")
    test_dataloader = DataLoader(test_data, batch_size=1)
    model, path = train(epochs=3, mtl=True)

    model.eval()
    preds = {"p_pred": [], "u_pred": [], "v_pred": []}
    for idx, s in enumerate(test_dataloader):
        #Get components
        x = s["x"]
        y = s["y"]
        t = s["t"]
        p = s["p"]
        u = s["u"]
        v = s["v"]

        #Make forward pass and backprop
        pred = model.forward(x, y, t, p, u, v)
        preds["p_pred"].extend(pred["p_pred"].detach().cpu().numpy())
        preds["u_pred"].extend(pred["u_pred"].detach().cpu().numpy())
        preds["v_pred"].extend(pred["v_pred"].detach().cpu().numpy())

    calculate_error(path, test_data, preds, model.lambda1.item(), model.lambda2.item())

