import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# Constants
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 添加用于年龄转换的one-hot编码函数
def one_hot_encode(age, num_classes=116):
    one_hot = torch.zeros(num_classes)
    one_hot[age - 1] = 1
    return one_hot

# 定义生成器类
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # 输入: 潜在向量 (z) 和年龄标签 (y)
            nn.ConvTranspose2d(100 + 116, 512, 4, 1, 0, bias=False),  # 使用116类的one-hot编码替换1
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
        y = y.unsqueeze(2).unsqueeze(3)  # 将年龄的one-hot编码调整为与噪声向量的形状匹配
        x = torch.cat([z, y], 1)
        return self.main(x)

# 加载预训练的生成器模型
generator_path = 'generator1_epoch_100.pth'
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(generator_path, map_location=DEVICE))

def generate_aged_face(input_image_path, desired_age, device):
    # 1. 打开输入图像并转换为RGB
    input_image = Image.open(input_image_path).convert('RGB')
    input_image_transformed = transforms.ToTensor()(input_image).unsqueeze(0).to(device)

    # 2. 生成目标年龄的one-hot编码
    target_age_one_hot = one_hot_encode(desired_age).unsqueeze(0).to(device)

    # 3. 使用预训练的生成器生成年龄化图像
    with torch.no_grad():
        z = torch.randn(1, 100, 1, 1).to(device)  # 生成随机潜在向量
        aged_image_tensor = generator(z, target_age_one_hot)

    # 4. 将生成的张量转换回PIL图像
    aged_image_tensor = (aged_image_tensor.squeeze(0) * 0.5 + 0.5).clamp(0, 1).detach().cpu()
    aged_image = transforms.ToPILImage()(aged_image_tensor)

    return aged_image

def save_image(image, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    image.save(os.path.join(output_dir, filename))


# 创建tkinter窗口
window = tk.Tk()
window.title("Age Progression")

frame = tk.Frame(window)
frame.pack(padx=20, pady=20)

# 添加年龄输入字段
age_label = tk.Label(frame, text="目标年龄:")
age_label.grid(row=0, column=0)
age_entry = tk.Entry(frame)
age_entry.grid(row=0, column=1)

# 添加“生成”按钮
def on_generate_button_click():
    desired_age = int(age_entry.get())
    input_image_path = filedialog.askopenfilename(title="选择输入图像")
    aged_image = generate_aged_face(input_image_path, desired_age, DEVICE)
    aged_image.show()

    # 保存生成的图像
    output_dir = filedialog.askdirectory(title="选择输出目录")
    filename = "aged_image.jpg"
    save_image(aged_image, output_dir, filename)
generate_button = tk.Button(frame, text="生成", command=on_generate_button_click)
generate_button.grid(row=1, column=0, columnspan=2)

window.mainloop()
