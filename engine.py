"""DeepLearning Framework module"""

from typing import Tuple, Dict, List
import torch

from tqdm.auto import tqdm 

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               trainloader: torch.utils.data.DataLoader,
               schedular: torch.optim.lr_scheduler=None,
               device: torch.device=DEVICE) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Args:
            model: A PyTorch model to be trained.
            dataloader: A DataLoader instance for the model to be trained on.
            loss_fn: A PyTorch loss function to minimize.
            optimizer: A PyTorch optimizer to help minimize the loss function.
            device: A target device to compute on (e.g. "cuda" or "cpu").

        Returns:
            A tuple of training loss and training accuracy metrics.
            In the form (train_loss, train_accuracy). For example:

        (0.1112, 0.8743)
    """ 
    train_loss, train_acc = 0, 0
    model.to(device)
    for _, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        model.train()
        y_logits = model(x)
        loss = loss_fn(y_logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if schedular is not None:
            schedular.step()

        y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        acc = torch.eq(y, y_pred).sum().item() / len(y)

        train_loss += loss
        train_acc += acc

    train_loss = train_loss / len(trainloader)
    train_acc = train_acc / len(trainloader)

    return train_loss, train_acc


def test_step(model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               testloader: torch.utils.data.DataLoader,
               device: torch.device=DEVICE) -> Tuple[float, float]:
    """
    Test step function, use to test the model on test data

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    test_loss, test_acc = 0, 0
    model.to(device)
    for _, (x, y) in enumerate(testloader):
        x, y = x.to(device), y.to(device)
        model.eval()
        with torch.inference_mode():
            y_logits = model(x)
            loss = loss_fn(y_logits, y)

            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            acc = torch.eq(y, y_pred).sum().item() / len(y)

            test_loss += loss
            test_acc += acc

    test_loss = test_loss / len(testloader)
    test_acc = test_acc / len(testloader)

    return test_loss, test_acc


def train(model: torch.nn.Module,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          trainloader: torch.utils.data.DataLoader,
          testloader: torch.utils.data.DataLoader,
          schedular: torch.optim.lr_scheduler=None,
          epochs: int=5,
          device: torch.device=DEVICE)-> Dict[str, List]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
        model: A PyTorch model to be trained and tested.
        train_dataloader: A DataLoader instance for the model to be trained on.
        test_dataloader: A DataLoader instance for the model to be tested on.
        optimizer: A PyTorch optimizer to help minimize the loss function.
        loss_fn: A PyTorch loss function to calculate loss on both datasets.
        epochs: An integer indicating how many epochs to train for.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A dictionary of training and testing loss as well as training and
        testing accuracy metrics. Each metric has a value in a list for 
        each epoch.
        In the form: {train_loss: [...],
                    train_acc: [...],
                    test_loss: [...],
                    test_acc: [...]} 
        For example if training for epochs=2: 
                    {train_loss: [2.0616, 1.0537],
                    train_acc: [0.3945, 0.3945],
                    test_loss: [1.2641, 1.5706],
                    test_acc: [0.3400, 0.2973]} 
    """
    result = {'train_loss':[],
            'train_acc':[],
            'test_loss':[],
            'test_acc':[]
            }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                        loss_fn=loss_fn,
                                        optimizer=optimizer,
                                        trainloader=trainloader,
                                        device=device)

        test_loss, test_acc = test_step(model=model,
                                    loss_fn=loss_fn,
                                    testloader=testloader,
                                    device=device)

        print(f'Epoch number: {epoch+1} | train loss: {train_loss:.4f} | test loss; {test_loss:.4f} | train accuracy: {train_acc:.4f} | test_acc: {test_acc:.4f}')

        result['train_loss'].append(train_loss.item())
        result['train_acc'].append(train_acc)
        result['test_loss'].append(test_loss.item())
        result['test_acc'].append(test_acc)

    return result
