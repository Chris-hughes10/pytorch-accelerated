import torch
from torch import nn


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(20, 10)
        self.head = nn.Linear(10, 5)
        # self.nested_head = nn.Sequential(nn.Linear(10, 10),
        #                                  nn.Sequential(
        #                                      nn.ReLU(),
        #                                      nn.Linear(10, 5),
        #                                  ),
        #                                  )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)


if __name__ == "__main__":
    model = SimpleModel()

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    head_p_value = model.head.weight
    backbone_p_value = model.backbone.weight

    x = torch.rand(1, 20)
    for i in range(2):
        preds = model(x)
        loss = loss_func(preds, torch.rand(1, 5))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # unfreeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = True
    optimizer.add_param_group({"params": model.backbone.parameters(), "lr": 0.01})

    for i in range(2):
        preds = model(x)
        loss = loss_func(preds, torch.rand(1, 5))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
