from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases




#TODO Get class weights
def getClassWeights():
    raise NotImplementedError

BATCH_SIZE = 16
TRANSFORM_PROBABILLITY = 1

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std),

        standard_transforms.RandomChoice([
        standard_transforms.RandomVerticalFlip(p=1),
        standard_transforms.RandomRotation(degrees=45),
        standard_transforms.RandomHorizontalFlip(p=1),
        standard_transforms.RandomCrop(size=(224,224)),], p=[TRANSFORM_PROBABILLITY for i in range(4)]),

        standard_transforms.RandomChoice([
        standard_transforms.RandomVerticalFlip(),
        standard_transforms.RandomRotation(degrees=45),
        standard_transforms.RandomHorizontalFlip(),
        standard_transforms.RandomCrop(size=(224,224))], p=[TRANSFORM_PROBABILLITY for i in range(4)])

    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size= BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= BATCH_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= BATCH_SIZE, shuffle=False)

epochs =10

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = 'cuda' if torch.cuda.is_available() else 'cpu' # TODO determine which device to use (cuda or cpu)
# device = 'mps' if torch.backends.mps.is_available() else 'cpu' # TODO determine which device to use (cuda or cpu)

# optimizer = torch.optim.SGD(fcn_model.parameters(), lr=0.01, momentum=0.9)
optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss() # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

fcn_model = fcn_model.to(device=device) # TODO transfer the model to the device


# TODO
def train():
    best_iou_score = 0.0

    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = fcn_model.forward(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score = val(epoch)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            # save the best model
    
 #TODO
def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(val_loader):
            input = input.to(device)
            output= fcn_model.forward(input)

            output = output.to('cpu')
            loss = criterion(output, label)
            losses.append(loss.item())

            pred= output.argmax(dim=1)
            mean_iou_scores.append(util.iou(pred, label))
            accuracy.append(util.pixel_acc(pred, label))




    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

 #TODO
def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !



    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            #TODO
            pass



    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!



if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()

    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()