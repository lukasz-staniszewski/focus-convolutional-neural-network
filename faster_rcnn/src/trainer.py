import os

import torch
from external.engine import evaluate, train_one_epoch


def save_model(model, model_name_prefix, epoch):
    model_path = os.path.join(
        "models", f"{model_name_prefix}_epoch_{epoch}.pth"
    )
    torch.save(model.state_dict(), model_path)


def train(
    model,
    dl_train,
    dl_validate,
    model_name_prefix,
    lr=0.005,
    n_epochs=10,
    print_freq=100,
):
    # model object
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=lr, momentum=0.9, weight_decay=0.005
    )

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=3, gamma=0.1
    # )

    for epoch in range(n_epochs):
        # train for one epoch, printing every 100 iterations
        train_one_epoch(
            model, optimizer, dl_train, device, epoch, print_freq=print_freq
        )

        # update the learning rate
        # lr_scheduler.step()

        save_model(model, model_name_prefix, epoch)

        # evaluate on the validation dataset
        evaluate(model, dl_validate, device=device)
