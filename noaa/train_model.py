
import random
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
from dataset_noaa import NOAADataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm


def get_model(device):
	# initialize the LeNet model
	print("[INFO] initializing the ResNet50 model...")
	model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
	num_ftrs = model.fc.in_features
	model.fc = torch.nn.Linear(num_ftrs, 5)  # there is 5 classes
	model = model.to(device)

	return model


def eval(val_dir, model, device):
	preprocess = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	val_ds = NOAADataset(val_dir, transforms=preprocess)
	batch_size = 16
	val_dl = DataLoader(val_ds, batch_size, pin_memory=True)

	lossFn = torch.nn.MSELoss()
	mae = torch.nn.L1Loss()
	total_val_loss = 0.
	mean_absolute_loss_per_class = np.array([0., 0., 0., 0., 0.])

	# switch off autograd for evaluation
	with torch.no_grad():
		# set the model in evaluation mode
		model.eval()
		# loop over the validation set
		for (x, y) in val_dl:
			# make the predictions and calculate the validation loss
			pred = model(x.to(device))
			y = y.to(device)
			total_val_loss += lossFn(pred, y)

			for i in range(5):
				mean_absolute_loss_per_class[i] += mae(pred[:, i], y[:, i])

	return total_val_loss.item() / len(val_dl), mean_absolute_loss_per_class / len(val_dl)


def train(train_dir, model, epochs, device):

	preprocess_and_augmentation = transforms.Compose([
		transforms.Resize(224),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomVerticalFlip(p=0.5)
	])

	# load the train and test data
	train_ds = NOAADataset(train_dir, transforms=preprocess_and_augmentation)

	batch_size = 16

	# load the train and validation into batches.
	train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)

	# initialize our optimizer and loss function
	opt = torch.optim.Adam(model.parameters(), lr=0.0001)
	lossFn = torch.nn.MSELoss() # it is an regression so we use MSE Loss
	# initialize a dictionary to store training history
	H = {
		"train_loss": [],
		"test_loss": [],
	}

	# loop over our epochs
	model.train()
	with tqdm(range(epochs), unit="epochs") as epochs:
		for e in epochs:

			# initialize the total training and validation loss
			total_train_loss = 0

			val_index = random.randint(0, len(train_dl) - 1)
			for i, (x, y) in enumerate(train_dl):

				if i != val_index:

					opt.zero_grad()
					# perform a forward pass and calculate the training loss
					pred = model(x.to(device))
					loss = lossFn(pred, y.to(device))
					# and update the weights
					loss.backward()
					opt.step()
					# add the loss to the total training loss so far and
					# calculate the number of correct predictions
					total_train_loss += loss.item()

				else:

					with torch.no_grad():
						pred = model(x.to(device))
						test_loss = lossFn(pred, y.to(device)).item()

			epochs.set_postfix(loss=total_train_loss / (len(train_dl) - 1), test_loss=test_loss)

			H["test_loss"].append(test_loss)
			H["train_loss"].append(total_train_loss / (len(train_dl) - 1))

	return H, model
