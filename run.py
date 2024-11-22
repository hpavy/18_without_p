from deepxrte.geometry import Rectangle
import torch
from utils import read_csv, write_csv, charge_data, init_model
from train import train
from pathlib import Path
import time
import pandas as pd
import numpy as np
import json


class RunSimulation():
    def __init__(self, hyper_param, folder_result_name, param_adim):
        self.hyper_param=hyper_param
        self.time_start = time.time()
        self.folder_result_name = folder_result_name
        self.folder_result = "results/" + folder_result_name
        # test seed, keep the same to compare the results
        self.random_seed_test = 2002
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.param_adim = param_adim
    
    def run(self):
        # Charging the model
        Path(self.folder_result).mkdir(parents=True, exist_ok=True)  # Creation du dossier de result
        if not Path(self.folder_result + "/hyper_param.json").exists():
            with open(self.folder_result + "/hyper_param.json", "w") as file:
                json.dump(self.hyper_param, file, indent=4)
            self.hyper_param = self.hyper_param

        else:
            with open(self.folder_result + "/hyper_param.json", "r") as file:
                self.hyper_param = json.load(file)

        ##### The code ###############################
        ###############################################

        # Data loading
        X_train_np, U_train_np, X_full, U_full, mean_std = charge_data(self.hyper_param, self.param_adim)
        X_train = torch.from_numpy(X_train_np).requires_grad_().to(torch.float32).to(self.device)
        U_train = torch.from_numpy(U_train_np).requires_grad_().to(torch.float32).to(self.device)


        # le domaine de résolution
        rectangle = Rectangle(
            x_max=X_full[:, 0].max(),
            y_max=X_full[:, 1].max(),
            t_min=X_full[:, 2].min(),
            t_max=X_full[:, 2].max(),
            x_min=X_full[:, 0].min(),
            y_min=X_full[:, 1].min(),
        )


        X_pde = rectangle.generate_lhs(self.hyper_param["nb_points_pde"]).to(self.device)

        # Data test loading
        torch.manual_seed(self.random_seed_test)
        np.random.seed(self.random_seed_test)
        X_test_pde = rectangle.generate_lhs(self.hyper_param["n_pde_test"]).to(self.device)
        points_coloc_test = np.random.choice(
            len(X_full), self.hyper_param["n_data_test"], replace=False
        )
        X_test_data = torch.from_numpy(X_full[points_coloc_test]).to(self.device)
        U_test_data = torch.from_numpy(U_full[points_coloc_test]).to(self.device)


        # Initialiser le modèle


        # On plot les print dans un fichier texte
        with open(self.folder_result + "/print.txt", "a") as f:
            model, optimizer, scheduler, loss, train_loss, test_loss = init_model(
                f, self.hyper_param, self.device, self.folder_result
            )
            ######## On entraine le modèle
            ###############################################
            train(
                nb_epoch=self.hyper_param["nb_epoch"],
                train_loss=train_loss,
                test_loss=test_loss,
                poids=[self.hyper_param["weight_data"], self.hyper_param["weight_pde"]],
                model=model,
                loss=loss,
                optimizer=optimizer,
                X_train=X_train,
                U_train=U_train,
                X_pde=X_pde,
                X_test_pde=X_test_pde,
                X_test_data=X_test_data,
                U_test_data=U_test_data,
                Re=self.hyper_param["Re"],
                time_start=self.time_start,
                f=f,
                u_mean=mean_std["u_mean"],
                v_mean=mean_std["v_mean"],
                x_std=mean_std["x_std"],
                y_std=mean_std["y_std"],
                t_std=mean_std["t_std"],
                u_std=mean_std["u_std"],
                v_std=mean_std["v_std"],
                p_std=mean_std["p_std"],
                folder_result=self.folder_result,
                save_rate=self.hyper_param["save_rate"],
                batch_size=self.hyper_param["batch_size"],
                scheduler=scheduler,
            )

        ####### On save le model et les losses

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            self.folder_result + "/model_weights.pth",
        )
        write_csv(train_loss, self.folder_result, file_name="/train_loss.csv")
        write_csv(test_loss, self.folder_result, file_name="/test_loss.csv")
