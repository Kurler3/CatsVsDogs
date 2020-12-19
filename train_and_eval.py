import torch
import torch.nn as nn
import torchvision
import time


def train(model, epochs, data_loader, optimizer, loss_func, model_save_location):
    since = time.time()
    model.train()
    train_size = len(data_loader)

    for epoch in range(epochs):
        running_loss = 0
        batch_loss = 0
        for i, (image, label) in enumerate(data_loader):
            batch_size = image.size(0)
            output = model(image)
            loss = loss_func(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * image.size(0)
            batch_loss += loss.item() * image.size(0)

            if i + 1 % batch_size == 0:
                time_batch = time.time() - since
                batch_loss /= batch_size
                print(
                    f'Epoch:{epoch + 1}/{epochs}, Batch Loss:{running_loss}, Time:{time_batch // 60} minutes {time_batch % 60} seconds')
                batch_loss = 0

        epoch_loss = running_loss / train_size

        time_epoch = time.time() - since
        print(f'Epoch:{epoch}/{epochs}, Epoch Loss:{epoch_loss}')

    # Training is over
    time_training = time.time() - since
    print(f'Finish Training in:{time_training // 60} minutes:{time_training % 60} seconds')

    try:
        torch.save(model.state_dict(), model_save_location)
    except:
        print("Something wrong happened when saving the model. Check if you saving location exists.")


def eval(model, eval_dataloader):
    since = time.time()

    total_images = 0
    total_correct = 0

    model.eval()
    with torch.no_grad():
        for image, label in eval_dataloader:
            output = model(image)

            _, predicts = torch.max(output.data, 1)

            total_images += image.size(0)
            total_correct += torch.sum(predicts == label.data)

    time_spent = time.time() - since
    percent_acc = 100 * total_correct / total_images
    print(f'Finished Evaluating in:{time_spent // 60} minutes:{time_spent % 60} seconds')
    print('Model Prediction Accuracy:{:.2f}%'.format(percent_acc))
