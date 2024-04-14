#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm


# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 62)  # 62 classes for EMNIST ByClass dataset

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # print(x.shape)
        x = x.view(-1, 128 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Load the trained model
model = CNN()
model.load_state_dict(torch.load('C:/Users/Seth Win10/Desktop/Brad AI Class/MNIST Project/hand_dataset_model/model.pt'))
model.eval()

# Define a preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))  # Assuming MNIST normalization values
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Inference function
def predict(image):
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

def main():
    st.title('Handwritten Letter Recognition')

    uploaded_file = st.file_uploader("Upload an image...", type=["png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        image = preprocess_image(image)
        prediction = predict(image)
        
        # Assuming the labels are numeric, map to corresponding letters
        label_mapping = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        predicted_letter = label_mapping[prediction]
        
        st.write(f'Predicted letter: {predicted_letter}')

if __name__ == "__main__":
    main()


# In[ ]:




