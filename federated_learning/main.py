import argparse
import shutil
import os

import numpy as np
import setGPU
import torch
import torchvision.transforms as transforms
import yaml

import fed.server
from fed.client import Client, MalClient
from datum.fmnist import FedFMNIST

np.random.seed(100)
torch.manual_seed(100)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError("No such file: {}".format(args.config))

    with open(args.config, "r") as f:
        print("The configuration file:")
        print(f.read())
        f.seek(0)
        config = yaml.safe_load(f)

    saved_path = config["saved_path"]
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    num_clients = config["num_clients"]

    # initialize clients and server                                                                                                                     [6/659]
    clients = list()
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = FedFMNIST("./datum/fmnist",
                          num_clients,
                          transform=transform,
                          evaluation=True)
    for i in range(num_clients):
        train_data = FedFMNIST("./datum/fmnist",
                               num_clients,
                               i,
                               transform=transform)
        client = Client(i, train_data, test_data, config)
        clients.append(client)

    saved_path = config["saved_path"]
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    num_clients = config["num_clients"]

    # initialize clients and server
    clients = list()
    transform = transforms.Compose([transforms.ToTensor()])
    test_data = FedFMNIST("./datum/fmnist",
                          num_clients,
                          transform=transform,
                          evaluation=True)
    for i in range(num_clients):
        train_data = FedFMNIST("./datum/fmnist",
                               num_clients,
                               i,
                               transform=transform)
        client = Client(i, train_data, test_data, config)
        clients.append(client)
    server = fed.server.Server(clients, test_data, config)

    # start federated learning
    server.run()
    server.dump_log()

    # copy config file to saved path
    shutil.copy2(args.config, saved_path)


if __name__ == "__main__":
    main()
