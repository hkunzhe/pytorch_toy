import copy
import os
from collections import OrderedDict, defaultdict

import torch
import torch.nn.functional as F
import yaml

from models.cnn import FourLayerNet
from .utils import compute_norm


class Server:
    def __init__(self, clients, test_data, config):
        self.clients = clients
        self.test_data = test_data
        self.config = config
        self.gmodel = FourLayerNet()

        self.log = defaultdict(list)
        self.device = torch.device("cuda" if self.config["device"] ==
                                   "cuda" else "cpu")

    def __broadcast(self):
        """Broadcast current global model to clients.
        """
        for client in self.clients:
            client.gmodel = copy.deepcopy(self.gmodel)

    @classmethod
    def __weight_average(cls, aggregate, num_clients):
        """The server collects `aggregate` update of clients, and
        compute an averaged update according to `FedAvg`.

        Args:
            aggregate (list): [client update], where the client
            update is OrderDict.
            num_clients (int): number of (selected) clients.

        Returns:
            OrderDict: {"layer name": averaged layer update}
        """
        averaged_update = OrderedDict()
        for k in aggregate[0].keys():
            averaged_update[k] = sum(update[k]
                                     for update in aggregate) / num_clients
        return averaged_update

    def __test(self, gmodel, test_loader):
        """Test the performance of `gmodel`.

        Args:
            gmodel (nn.Module): global model.
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            tuple: (averaged test loss, accuracy)
        """
        gmodel.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.long().to(
                    self.device)
                output = gmodel(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)

        return test_loss, acc

    def __update(self, update):
        """update global model by add `update` in-place.

        Args:
            update (OrderDict): update.
        """
        for param1, param2 in zip(self.gmodel.parameters(), update.values()):
            param1.data += param2.data

    def run(self):
        total_rounds = self.config["total_rounds"]
        num_clients = self.config["num_clients"]
        test_loader = torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=len(
                                                      self.test_data))
        self.gmodel.to(self.device)
        for r in range(total_rounds):
            print("====================Round {}====================".format(r))
            self.__broadcast()
            aggregation = list()
            for client in self.clients:
                update = client.run()
                client.cur_round += 1
                aggregation.append(update)
            averaged_update = self.__weight_average(aggregation, num_clients)
            self.log["aggregated_norm"].append(
                compute_norm(averaged_update).item())
            self.__update(averaged_update)

            test_loss, acc = self.__test(self.gmodel, test_loader)
            self.log["test_loss"].append(test_loss)
            self.log["accuracy"].append(acc)

            # save global model
            model_name = "server_" + "round" + str(r) + ".pt"
            model_dir = os.path.join(self.config["saved_path"], "models")
            model_path = os.path.join(model_dir, model_name)
            torch.save(self.gmodel.state_dict(), model_path)

    def dump_log(self):
        """Dump logs for server and clients when federated learning finished.
        """
        for client in self.clients:
            client.dump_log()

        log_name = "server_" + "log" + ".yaml"
        log_dir = os.path.join(self.config["saved_path"], "logs")
        with open(os.path.join(log_dir, log_name), "w") as f:
            yaml.dump(self.log, f)

