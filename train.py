import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Constants
BATCH_SIZE = 500
IMAGE_SIZE = 64
LEARNING_RATE = 0.0002
NUM_EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 使用GPU（如果可用）或CPU

if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

# 添加一个one_hot编码函数，用于将年龄转换为one_hot编码
def one_hot_encode(age, num_classes=116):
    one_hot = torch.zeros(num_classes)
    one_hot[age - 1] = 1
    return one_hot

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # Input: latent vector (z) and age label (y)
            nn.ConvTranspose2d(100 + 116, 512, 4, 1, 0, bias=False),  # 使用116类的one_hot编码替换1
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # Layer 1
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Layer 2
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Layer 3
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Layer 4
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y):
        y = y.unsqueeze(2).unsqueeze(3)  # 将年龄的one_hot编码调整为与噪声向量的形状匹配
        x = torch.cat([z, y], 1)
        return self.main(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # Layer 0
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 1
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.age_classifier = nn.Sequential(
            # Output
            nn.Conv2d(512 + 116, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        features = self.main(x)
        y = y.view(y.size(0), -1, 1, 1).repeat(1, 1, 4, 4)  # 修改这一行
        features = torch.cat([features, y], 1)
        age_output = self.age_classifier(features)
        return age_output.squeeze()


# UTKFace Dataset
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.file_list[idx])
        image = Image.open(img_name).convert('RGB')

        # Extract age from filename
        age = int(self.file_list[idx].split('_')[0])

        # Convert age to one_hot encoding
        one_hot_age = np.zeros(116)
        one_hot_age[age - 1] = 1
        one_hot_age = torch.tensor(one_hot_age, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, one_hot_age


# Create Dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = UTKFaceDataset(root_dir='UTKface_inthewild/part1', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create models and optimizers
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))

criterion = nn.BCELoss()

# Training loop
for epoch in range(NUM_EPOCHS):
    for i, (real_images, real_ages) in enumerate(train_loader):
        real_images = real_images.to(DEVICE)
        real_ages = real_ages.float().to(DEVICE)

        # Update Discriminator
        optimizer_D.zero_grad()

        z = torch.randn(real_images.size(0), 100, 1, 1).to(DEVICE)
        fake_images = generator(z, real_ages)
        real_output = discriminator(real_images, real_ages)
        fake_output = discriminator(fake_images.detach(), real_ages)

        real_loss = criterion(real_output, torch.ones_like(real_output))
        fake_loss = criterion(fake_output, torch.zeros_like(fake_output))
        d_loss = real_loss + fake_loss

        d_loss.backward()
        optimizer_D.step()

        # Update Generator
        optimizer_G.zero_grad()

        fake_output = discriminator(fake_images, real_ages)
        g_loss = criterion(fake_output, torch.ones_like(fake_output))

        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch: {epoch + 1}/{NUM_EPOCHS}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")


# Save models
if (epoch + 1) % 100 == 0:
    torch.save(generator.state_dict(), f'generator1_epoch_{epoch + 1}.pth')
    torch.save(discriminator.state_dict(), f'discriminator1_epoch_{epoch + 1}.pth')
