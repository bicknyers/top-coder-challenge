import time
import lightning as pl
import torch
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from torch.utils.data import DataLoader, random_split
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch import loggers as pl_loggers
import torch.nn.functional as F

# from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt

# import sklearn
# from pyperch.neural.ga_nn import GAModule
# from pyperch.neural.rhc_nn import RHCModule
# from pyperch.neural.sa_nn import SAModule
# from skorch import NeuralNetClassifier
# from skorch.callbacks import EpochScoring, TensorBoard


class JsonDataset(Dataset):
    def __init__(self, json_file, norm=True):
        self.json_file = json_file
        self.norm = norm
        with open(self.json_file, 'r') as f:
            self.data = json.load(f)
        self.out = []
        self.miles = []
        self.rec = []
        self.dur = []

        for case in self.data:
            self.out.append(case['expected_output'])
            self.miles.append(case['input']['miles_traveled'])
            self.rec.append(case['input']['total_receipts_amount'])
            self.dur.append(case['input']['trip_duration_days'])

        self.out = np.array(self.out)
        self.miles = np.array(self.miles)
        self.rec = np.array(self.rec)
        self.dur = np.array(self.dur)

        self.out_norm_weight = np.max(self.out)-np.min(self.out)
        self.out_norm = self.out / self.out_norm_weight
        self.out_norm_bias = np.min(self.out_norm)
        self.out_norm = self.out_norm - self.out_norm_bias

        self.miles_norm_weight = np.max(self.miles)-np.min(self.miles)
        self.miles_norm = self.miles / self.miles_norm_weight
        self.miles_norm_bias = np.min(self.miles_norm)
        self.miles_norm = self.miles_norm - self.miles_norm_bias

        self.rec_norm_weight = np.max(self.rec)-np.min(self.rec)
        self.rec_norm = self.rec / self.rec_norm_weight
        self.rec_norm_bias = np.min(self.rec_norm)
        self.rec_norm = self.rec_norm - self.rec_norm_bias

        self.dur_norm_weight = np.max(self.dur)-np.min(self.dur)
        self.dur_norm = self.dur / self.dur_norm_weight
        self.dur_norm_bias = np.min(self.dur_norm)
        self.dur_norm = self.dur_norm - self.dur_norm_bias

        self.norm_in_bias = torch.from_numpy(np.stack([self.miles_norm_bias, self.rec_norm_bias, self.dur_norm_bias])).to(dtype=torch.float32)
        self.norm_in_weight = torch.from_numpy(np.stack([self.miles_norm_weight, self.rec_norm_weight, self.dur_norm_weight])).to(dtype=torch.float32)

        self.norm_out_bias = torch.from_numpy(np.stack([self.out_norm_bias])).to(dtype=torch.float32)
        self.norm_out_weight = torch.from_numpy(np.stack([self.out_norm_weight])).to(dtype=torch.float32)

        return

    def __len__(self):
        return len(self.out)

    def __getitem__(self, idx):
        if self.norm:
            return torch.from_numpy(np.stack([self.miles_norm[idx], self.rec_norm[idx], self.dur_norm[idx]])).to(dtype=torch.float32), torch.from_numpy(np.stack([self.out_norm[idx]])).to(torch.float32)
        else:
            return torch.from_numpy(np.stack([self.miles[idx], self.rec[idx], self.dur[idx]])).to(dtype=torch.float32), torch.from_numpy(np.stack([self.out[idx]])).to(torch.float32)


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight, nonlinearity='linear')
        if m.bias is not None:
            init.zeros_(m.bias)


class LitClassifier(pl.LightningModule):
    def __init__(self, norm_in_weight, norm_in_bias, norm_out_weight, norm_out_bias, num_features=3, hidden_dim1=16, hidden_dim2=8, hidden_dim3=4, learning_rate=0.00005):

        super().__init__()
        self.save_hyperparameters()

        self.loss_fn = torch.nn.MSELoss()

        self.norm_in = torch.nn.Linear(1, num_features, bias=True)
        self.l1 = torch.nn.Linear(num_features, hidden_dim1, bias=True)
        self.l2 = torch.nn.Linear(hidden_dim1, hidden_dim2, bias=True)
        self.l3 = torch.nn.Linear(hidden_dim2, hidden_dim3, bias=True)
        self.l4 = torch.nn.Linear(hidden_dim3, 1, bias=True)
        self.norm_out = torch.nn.Linear(1, 1, bias=True)

        self.apply(_weights_init)

        # Set normalize weights and freeze
        self.norm_in.weight.data = norm_in_weight
        self.norm_in.weight.requires_grad = False
        self.norm_in.bias.data = norm_in_bias
        self.norm_in.bias.requires_grad = False

        self.norm_out.weight.data = 1/norm_out_weight
        self.norm_out.weight.requires_grad = False
        self.norm_out.bias.data = -norm_out_bias
        self.norm_out.bias.requires_grad = False


    def forward(self, x):
        # x = x.view(x.size(0), -1)
        # x = self.selu1(self.l1(x))
        # x = self.selu2(self.l2(x))
        # x = self.l3(x)
        #
        # if self.use_sig_out:
        #     x = self.sig(x)

        return F.relu(self.l4(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x))))))))


    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log('loss/train', loss)

        return loss


    # def validation_step(self, batch, batch_idx):
    #     x, y = batch
    #     preds = self(x)
    #     loss = self.loss_fn(preds, y)
    #
    #     self.log('loss/validation', loss)
    #
    #     return loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        loss = self.loss_fn(preds, y)

        self.log('loss/test', loss)

        return loss


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00001)
        warmup_sch = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / 10.0, end_factor=1.0,
                                                       total_iters=10)
        cosine_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, eta_min=0.000001)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_sch, cosine_sch], [10])
        return [optimizer], [scheduler]

def train(dataset_path, seed, batch_size, h1, h2, h3):
    start_time = time.time()

    pl.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('highest')

    dataset = JsonDataset(dataset_path, norm=True)
    # train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])
    train_dataset = dataset
    val_dataset = dataset
    test_dataset = dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=8, drop_last=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=8, drop_last=False, pin_memory=True)

    model = LitClassifier(dataset.norm_in_weight, dataset.norm_in_bias, dataset.norm_out_weight, dataset.norm_out_bias, hidden_dim1=h1, hidden_dim2=h2, hidden_dim3=h3)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="logs/")
    lr_monitor = LearningRateMonitor(logging_interval=None, log_momentum=True, log_weight_decay=True)

    trainer = pl.Trainer(max_epochs=30, logger=tb_logger, deterministic=True, precision="32-true", callbacks=[lr_monitor], log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)

    end_time = round(time.time() - start_time, 2)
    print("Time Elapsed: " + str(end_time))

    return


if __name__ == '__main__':
    batch_sizes = [32]
    h1s = [256]
    h2s = [1024]
    h3s = [256, 512]

    for batch_size_i in batch_sizes:
        for h1i in h1s:
            for h2i in h2s:
                for h3i in h3s:
                    train('public_cases.json', seed=42, batch_size=batch_size_i, h1=h1i, h2=h2i, h3=h3i)

