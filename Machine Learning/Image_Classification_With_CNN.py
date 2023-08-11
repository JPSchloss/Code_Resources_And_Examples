# Importing the necessary libraries
import torch                                    # This is the main PyTorch library for tensor computations and DL. 
import torch.nn as nn                           # This is the neural networks library in PyTorch.
import torch.nn.functional as F
import torchvision                              # This library is used for computer vision tasks and comes with PyTorch.
import torchvision.transforms as transforms     # Library with common image transformations. 
import torchvision.transforms.functional as TF  # Functional interface for transformers library. 
import torch.optim as optim                     # PyTorch package containing optimization algorithms. 
import matplotlib.pyplot as plt 
import numpy as np


# Defining the simple convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()       # Calling the parent constructor
        self.conv1 = nn.Conv2d(3, 6, 5)         # Defining the first convolutional layer
        self.pool = nn.MaxPool2d(2, 2)          # Defining max-pooling layer
        self.conv2 = nn.Conv2d(6, 16, 5)        # Defining the second convolutional layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   # Defining the first fully connected layer
        self.fc2 = nn.Linear(120, 84)           # Defining the second fully connected layer
        self.fc3 = nn.Linear(84, 10)            # Defining the third fully connected layer

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # Passing input through first conv layer, then activation, then pooling
        x = self.pool(F.relu(self.conv2(x)))    # Passing input through second conv layer, then activation, then pooling
        
        x = x.view(-1, 16 * 5 * 5)              # Flattening the tensor
        
        x = F.relu(self.fc1(x))                 # Passing through the first fully connected layer with activation
        x = F.relu(self.fc2(x))                 # Passing through the second fully connected layer with activation
        x = self.fc3(x)                         # Passing through the third fully connected layer
        
        return x

# Defining an image classifier class
class ImageClassifier:
    def __init__(self, model):
        self.model = model                      # Initializing the model
        self.criterion = nn.CrossEntropyLoss()  # Defining the loss function
        # Defining the optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, dataset):
        # Defining transformations for the dataset
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # Apply transformations to the dataset
        dataset.transform = transform
        # Creating a data loader for batching
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

        # Setting up the training loop
        for epoch in range(2):  
            running_loss = 0.0                  # Initialize running loss
        
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data                   # Unpacking the data to get inputs and labels
                self.optimizer.zero_grad()              # Reseting optimizer gradients
                
                outputs = self.model(inputs)            # Conducting forward pass
                loss = self.criterion(outputs, labels)  # Computing the loss
                loss.backward()                         # Backward pass to compute gradients
                self.optimizer.step()                   # Updating the model parameters

                running_loss += loss.item()             # Updating the running loss
                
                # Printing statistics after every 2000 batches
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def classify_image(self, image):
        image = TF.to_tensor(image).unsqueeze(0)        # Converting the image to tensor and adding a batch dimension
        outputs = self.model(image)                     # Getting the model predictions for the image
        _, predicted = torch.max(outputs, 1)            # Getting the index of the maximum value to determine class
        return predicted.item()

# Defining the main function. 
def main():
    # Loading in the CIFAR-10 training and test datasets
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)

    # Listing the class names in CIFAR-10
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiating the CNN model and image classifier
    net = SimpleCNN()
    classifier = ImageClassifier(net)
    
    # Training the classifier
    classifier.train(trainset)

    # Retrieving one image from the test dataset 
    image, label = testset[42]

    # Using the trained model to classify the image
    predicted = classifier.classify_image(image)
    
    # Printing the results
    print('GroundTruth: ', classes[label])
    print('Predicted: ', classes[predicted])

    # Displaying the image for visual confirmation
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    main()
