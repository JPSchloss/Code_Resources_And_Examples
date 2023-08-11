# Importing the necessary libraries
import torch                                    
import torch.nn as nn                           
import torch.nn.functional as F
import torchvision                              
import torchvision.transforms as transforms     
import torchvision.transforms.functional as TF  
import torch.optim as optim                     
import matplotlib.pyplot as plt 
import numpy as np


# Defining the simple convolutional neural network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()       
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)


    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# Defining an image classifier class
class ImageClassifier:
    def __init__(self, model):
        self.model = model                      
        self.criterion = nn.CrossEntropyLoss()  
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, dataset):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset.transform = transform
        trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

        # Setting up the training loop
        for epoch in range(2):  
            running_loss = 0.0                  
        
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data                   
                self.optimizer.zero_grad()              
                
                outputs = self.model(inputs)            
                loss = self.criterion(outputs, labels)  
                loss.backward()                         
                self.optimizer.step()                   

                running_loss += loss.item()             
                
                # Printing statistics after every 2000 batches
                if i % 2000 == 1999:
                    print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def classify_image(self, image):
        image = TF.to_tensor(image).unsqueeze(0)        
        outputs = self.model(image)                     
        _, predicted = torch.max(outputs, 1)           
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

    # Train the model
    train_model = True  # set this to False if you want to load the model instead
    
    if train_model:
        classifier.train(trainset)
        torch.save(classifier.model.state_dict(), 'model.pth')  # save the model
    else:
        classifier.model.load_state_dict(torch.load('model.pth'))  # load the model

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
