import copy
import os
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from tqdm import trange

from .utils import compute_norm


class Client():
    def __init__(self,
                 index,
                 train_data,
                 test_data,
                 config,
                 cur_round=0,
                 gmodel=None):
        self.index = index
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.gmodel = gmodel
        self.cur_round = cur_round

        self.log = defaultdict(list)
        self.device = torch.device("cuda" if config["device"] ==
                                   "cuda" else "cpu")

    def train(self, lmodel, train_loader):
        """Train `lmodel` locally at current round. Note `lmodel`
        should be put to specific device before local training.

        Args:
            lmodel (nn.Module): Local model.
            train_loader (DataLoader): DataLoader for training data.
        """
        epoches = self.config["local_epoches"]
        lr = self.config["local_lr"]

        optim = torch.optim.Adam(lmodel.parameters(), lr=lr)
        lmodel.train()
        for epoch in trange(epoches, desc="training", unit="epoch"):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.long().to(
                    self.device)
                optim.zero_grad()
                output = lmodel(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optim.step()

    def test(self, lmodel, test_loader):
        """Test the performance of `lmodel`.

        Args:
            lmodel (nn.Module): Local model.
            test_loader (DataLoader): DataLoader for test data.

        Returns:
            tuple: (averaged test loss, accuracy)
        """
        lmodel.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.long().to(
                    self.device)
                output = lmodel(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)

        return test_loss, acc

    @classmethod
    def compute_update(cls, lmodel, gmodel):
        """Compute model update during local training by `lmodel - gmodel` element-wise.

        Args:
            lmodel (nn.Module): Local model.
            gmodel (nn.Module): Global model.

        Returns:
            OrderDict: {"layer name": layer update}.
        """
        update = OrderedDict()
        for (name, param1), (_, param2) in zip(lmodel.named_parameters(),
                                               gmodel.named_parameters()):
            update[name] = param1.data - param2.data
        return update

    def run(self):
        batch_size = self.config["batch_size"]
        lmodel = copy.deepcopy(self.gmodel)
        lmodel = lmodel.to(self.device)
        train_loader = torch.utils.data.DataLoader(self.train_data,
                                                   batch_size=batch_size,
                                                   shuffle=True)
        test_loader = torch.utils.data.DataLoader(self.test_data,
                                                  batch_size=len(
                                                      self.test_data))
        print("Client {} start training with {} samples at round {}\n".format(
            self.index, len(self.train_data), self.cur_round))
        test_loss, acc = self.test(lmodel, test_loader)
        print(
            "Before local training, averaged test loss is {} and accuracy is {}%"
            .format(test_loss, acc))
        self.train(lmodel, train_loader)

        test_loss, acc = self.test(lmodel, test_loader)
        self.log["test loss"].append(test_loss)
        self.log["accuracy"].append(acc)
        print(
            "After local training, averaged test loss is {} and accuracy is {}%\n"
            .format(test_loss, acc))

        # save local model
        model_name = "client" + str(self.index) + "_round" + str(
            self.cur_round) + ".pt"
        model_dir = os.path.join(self.config["saved_path"], "models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(lmodel.state_dict(), os.path.join(model_dir, model_name))

        update = self.compute_update(lmodel, self.gmodel)
        self.log["local_model_norm"].append(compute_norm(lmodel).item())
        self.log["update_norm"].append(compute_norm(update).item())
        return update

    def dump_log(self):
        log_name = "client" + str(self.index) + "_log" + ".yaml"
        log_dir = os.path.join(self.config["saved_path"], "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, log_name), "w") as f:
            yaml.dump(self.log, f)

