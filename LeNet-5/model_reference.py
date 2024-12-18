import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
from cnn_model import CNN

class Identifier:
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.model = torch.load('zjj.pkl')
        self.model.eval()

    def identify_number(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (28, 28))
        _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0)
        with torch.no_grad():
            output = self.model(img)
            pred = torch.argmax(output, dim=1).item()
        return pred

if __name__ == '__main__':
    identifier = Identifier()
    print(identifier.identify_number('path/to/image.jpg'))  # 替换为实际路径
