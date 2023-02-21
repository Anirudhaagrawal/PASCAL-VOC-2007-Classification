from tqdm import tqdm

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


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


BATCH_SIZE = 16
TRANSFORM_PROBABILLITY = 0.1

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


# TODO
def train():
    best_iou_score = 0.0
    # plot data

    # weights = train_dataset.get_class_weights()
    # loading bar
    training_pbar = tqdm(total=epochs, desc=f'Training Procedure', position=0)
    train_size = len(train_loader.dataset)
    for epoch in range(epochs):
        inner_pbar = tqdm(total=train_size, desc=f'Training Epoch {epoch + 1}', position=0, leave=True)
        ts = time.time()
        iters = len(train_loader)
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            criterion = nn.CrossEntropyLoss(weight=class_weights)

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + iter / iters)
            inner_pbar.update(train_loader.batch_size)
        inner_pbar.close()
        current_mean_iou_score = val(epoch)
        if current_mean_iou_score > best_iou_score:
            best_iou_score = current_mean_iou_score
            # save the best model
        training_pbar.update(1)
    training_pbar.close()


 #TODO
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

    return np.mean(mean_iou_scores)


# TODO
def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            # TODO
            pass

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


if __name__ == "__main__":
    val(0)  # show the accuracy before training
    train()

    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
