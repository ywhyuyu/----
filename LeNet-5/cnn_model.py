from lenet5 import LeNet5

class CNN(LeNet5):  # 继承 LeNet5
    def __init__(self, num_classes=10):
        super(CNN, self).__init__(num_classes=num_classes)
