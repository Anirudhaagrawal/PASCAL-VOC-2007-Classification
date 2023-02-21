from tqdm import tqdm

import os
from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import torch.backends.mps as backends
from resnet import *
from unet import *


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


def init_weights_transfer_learning(m):
    if isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


BATCH_SIZE = 16
TRANSFORM_PROBABILLITY = 0.1
U_NET = False
Fcn = False
RESNET = True
epochs = 30

n_class = 21

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std),

])
target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
if U_NET:
    fcn_model = UNet(n_class=n_class)
    fcn_model.apply(init_weights)
elif RESNET:
    fcn_model = Resnet(n_class=n_class)
    fcn_model.apply(init_weights_transfer_learning)
elif Fcn:
    fcn_model = FCN(n_class=n_class)
    fcn_model.apply(init_weights)

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # TODO determine which device to use (cuda or cpu)

device = torch.device(device)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.001)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=1)
weights = train_dataset.get_class_weights()
class_weights = torch.FloatTensor(weights)
criterion = nn.CrossEntropyLoss(
    weight=class_weights)  # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html
class_weights = class_weights.to(device)

fcn_model = fcn_model.to(device=device)  # TODO transfer the model to the device


def train(save_location):

    # ---------------------------------
    # Initialize network, progress bar,
    # arrays to record train/val accuracy
    # and loss, counter of bad epochs
    # ---------------------------------
    best_iou_score = 0.0
    val_loss = np.zeros(epochs)
    train_loss = np.zeros(epochs)
    mean_iou_scores = np.zeros(epochs)
    val_accuracy = np.zeros(epochs)
    early_stop_patience = 2
    early_stop = True # flag to identify if early stopping is desired

    # number of consecutive epochs where
    # model performs worse
    bad_epochs = 0
    earlyStop = -1

    # weights = train_dataset.get_class_weights()
    # loading bar
    training_pbar = tqdm(total=epochs, desc=f'Training Procedure', position=0)
    train_size = len(train_loader.dataset)

    # ------------------
    # Training Procedure
    # ------------------
    for epoch in range(epochs):
        inner_pbar = tqdm(total=train_size, desc=f'Training Epoch {epoch + 1}', position=0, leave=True)
        ts = time.time()
        iters = len(train_loader)
        train_losses = []
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model.forward(inputs)

            loss = criterion(outputs, labels)
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step(epoch + iter / iters)

            inner_pbar.update(train_loader.batch_size)
        train_loss[epoch] = np.mean(train_losses)
        inner_pbar.close()
        
        current_miou_score, current_accuracy, current_val_loss = val(epoch)
        val_loss[epoch]=current_val_loss
        mean_iou_scores[epoch]=current_miou_score
        val_accuracy[epoch]=current_accuracy

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            path = save_location + 'model.pt'
            torch.save(fcn_model, path)
            # save the best model

        if epoch > 0 and early_stop and current_val_loss > val_loss[epoch - 1]:
            bad_epochs += 1
        else:
            bad_epochs = 0
        if bad_epochs > early_stop_patience:
            earlyStop = epoch
            print(f'Patience threshold reached ({early_stop_patience} epochs).')
            print(f'Early stopping after completing epoch {epoch + 1}.')
            break
        
        training_pbar.update(1)
    training_pbar.close()
    util.plots(train_loss, val_loss, val_accuracy, mean_iou_scores, earlyStop, saveLocation = save_location)

def val(epoch):
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []
    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        val_size = len(val_loader.dataset)
        val_pbar = tqdm(total=val_size, desc=f'Validation Epoch {epoch + 1}', position=0, leave=True)
        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device)
            output = fcn_model.forward(input)

            output = output.to('cpu')
            loss = criterion(output, label)
            losses.append(loss.item())

            pred = output.argmax(dim=1)
            mean_iou_scores.append(util.iou(pred, label))
            accuracy.append(util.pixel_acc(pred, label))
            val_pbar.update(val_loader.batch_size)
        val_pbar.close()
    tqdm.write(f'Epoch\t{epoch + 1}')
    tqdm.write(f"loss\t{np.mean(losses)}")
    tqdm.write(f"IoU\t{np.mean(mean_iou_scores)}")
    tqdm.write(f"PA\t{np.mean(accuracy)}")

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores), np.mean(accuracy), np.mean(losses)

def modelTest(save_location):
    path = save_location+'model.pt'
    model = torch.load(path)
    model.eval()
    #fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    i=0
    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            input = input.to(device)
            output = model.forward(input)

            output = output.to('cpu')
            loss = criterion(output, label)

            pred = output.argmax(dim=1)

            input = input.to('cpu')
            util.plot_predictions(input[0], label[0], pred[0], i)
            i = i + 1


if __name__ == "__main__":
    import os
    path = 'Results'
    if not os.path.exists(path):
      os.mkdir(path)
    save_location = path+"/model3a"
    val(0)  # show the accuracy before training
    train(save_location)
    modelTest(save_location)

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
