import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2

class model_wrapper:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.return_img = False
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        im_pil = Image.fromarray(img)

        img = self.preprocess(im_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img).cpu().numpy()[0]
        pred[pred < 0] = 0.
        pred = pred.astype(int)
        detected = np.all(pred > 0)

        return pred, detected


class windowed_search:

    def __init__(self, model):
        self.model = model
        self.img = None

    def load_img(self, img):
        self.img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        H, W, channels = self.img.shape
        # check which dimention is the bigger
        max_ = np.max([W, H])
        # check that the image is divisble by 2
        if max_ % 2:
            max_ += 1

        self.img = cv2.copyMakeBorder(self.img, 0, max_ - H, 0,
                                           max_ - W, cv2.BORDER_CONSTANT, None, value=0)

        self.dim = max_
        self.min_res = self.dim
        self.nb_zoom_max = 0
        while self.min_res / 2 > 224:
            self.nb_zoom_max += 1
            self.min_res /= 2

        self.nb_max_conv_action = self.dim / self.min_res

    def get_window(self, x, y):
        return self.img[y: y + int(self.min_res), x: x + int(self.min_res)]

    def __call__(self, img):

        self.load_img(img)

        preds = {}
        counter = 0
        while counter < self.nb_max_conv_action ** 2:
            y = int((counter / self.nb_max_conv_action) * self.min_res)
            x = int((counter % self.nb_max_conv_action) * self.min_res)
            sub_img = self.get_window(x, y)
            sub_img = cv2.resize(sub_img, (224, 224))

            if np.all(sub_img == 0):
                break

            pred, _ = self.model(sub_img)
            preds[(x, y, int(self.min_res))] = pred

            counter += 1

        return preds





