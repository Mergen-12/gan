import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from math import log2
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants
START_DIM = 512
MAPPING_LAYERS = 8
STYLE_DIM = 512
OUTPUT_CHANNELS = 3

# Utility functions
def load_dataset(dataset_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(root=dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)


class FaceDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform=None):
        self.dataset_path = dataset_path
        self.transform = transform
        self.image_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        return image

# Mapping Network
class MappingNetwork(nn.Module):
    def __init__(self):
        super(MappingNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(MAPPING_LAYERS):
            self.layers.append(nn.Linear(STYLE_DIM, STYLE_DIM))
            self.layers.append(nn.LeakyReLU(0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Spatial Adaptive Normalization (SPAN) Layer
class SPANLayer(nn.Module):
    def __init__(self, in_channels, style_dim):
        super(SPANLayer, self).__init__()
        self.norm = nn.InstanceNorm2d(in_channels, affine=False)
        self.style_scale = nn.Conv2d(style_dim, in_channels, 1, 1)
        self.style_shift = nn.Conv2d(style_dim, in_channels, 1, 1)

    def forward(self, x, style):
        x = self.norm(x)
        scale = self.style_scale(style)
        shift = self.style_shift(style)
        return x * scale + shift

# StyleGAN Generator
class StyleGANGenerator(nn.Module):
    def __init__(self, latent_dim, output_size=1024, landmark_dim=68*2):
        super(StyleGANGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.output_size = output_size
        self.landmark_dim = landmark_dim

        self.mapping = MappingNetwork()

        self.input = nn.Parameter(torch.randn(1, START_DIM, 4, 4))
        self.span1 = SPANLayer(START_DIM, STYLE_DIM)
        self.conv1 = nn.Conv2d(START_DIM, START_DIM, 3, padding=1)
        self.span2 = SPANLayer(START_DIM, STYLE_DIM)

        self.upsamples = nn.ModuleList()
        self.spans = nn.ModuleList()
        self.convs = nn.ModuleList()

        in_channels = START_DIM
        for _ in range(int(log2(output_size // 4))):
            out_channels = in_channels // 2
            self.upsamples.append(nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1))
            self.spans.append(SPANLayer(out_channels, STYLE_DIM))
            self.convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            in_channels = out_channels

        self.final_span = SPANLayer(out_channels, STYLE_DIM)
        self.final_conv = nn.Conv2d(out_channels, OUTPUT_CHANNELS, 1)

        self.landmark_fc = nn.Linear(landmark_dim, STYLE_DIM)

    def forward(self, latent_vector, landmarks):
        styles = self.mapping(latent_vector)
        landmark_styles = self.landmark_fc(landmarks)

        x = self.input.repeat(latent_vector.shape[0], 1, 1, 1)
        x = self.span1(x, styles + landmark_styles)
        x = self.conv1(x)
        x = self.span2(x, styles + landmark_styles)

        for upsample, span, conv in zip(self.upsamples, self.spans, self.convs):
            x = upsample(x)
            x = span(x, styles + landmark_styles)
            x = conv(x)

        x = self.final_span(x, styles + landmark_styles)
        x = self.final_conv(x)

        return x
    
# Discriminator network
class StyleGANDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(StyleGANDiscriminator, self).__init__()
        self.input_size = input_size

        self.conv1 = nn.Conv2d(OUTPUT_CHANNELS, START_DIM, 1)
        self.span1 = SPANLayer(START_DIM, STYLE_DIM)
        self.conv2 = nn.Conv2d(START_DIM, START_DIM, 3, padding=1)
        self.span2 = SPANLayer(START_DIM, STYLE_DIM)

        self.downsamples = nn.ModuleList()
        self.spans = nn.ModuleList()
        self.convs = nn.ModuleList()

        in_channels = START_DIM
        for _ in range(int(log2(input_size // 4))):
            out_channels = in_channels * 2
            self.downsamples.append(nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1))
            self.spans.append(SPANLayer(out_channels, STYLE_DIM))
            self.convs.append(nn.Conv2d(out_channels, out_channels, 3, padding=1))
            in_channels = out_channels

        self.fc = nn.Linear(out_channels * 4 * 4, 1)

    def forward(self, x, styles):
        x = self.conv1(x)
        x = self.span1(x, styles)
        x = self.conv2(x)
        x = self.span2(x, styles)

        for downsample, span, conv in zip(self.downsamples, self.spans, self.convs):
            x = downsample(x)
            x = span(x, styles)
            x = conv(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Parameters
latent_dim = 512
img_size = 1024
landmark_dim = 68 * 2
batch_size = 32
lr = 0.0001
epochs = 10

# Initialize the discriminator
discriminator = StyleGANDiscriminator(input_size=img_size).to(device)

# Training loop
def train_gan(generator, discriminator, dataloader, num_epochs, latent_dim, image_size):
    criterion = nn.BCEWithLogitsLoss()
    fixed_noise = torch.randn(64, latent_dim, device=device)

    real_label = 1
    fake_label = 0

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, latent_dim, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_D.step()

            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if i % 500 == 0:
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                save_image(fake, 'gan_images/fake_samples_epoch_%03d.png' % epoch, normalize=True)

# Initialize the generator
generator = StyleGANGenerator(latent_dim=latent_dim, output_size=img_size).to(device)
dataset = load_dataset("dataset/path", image_size=(img_size, img_size))
train_gan(generator, discriminator, dataset, num_epochs=epochs, latent_dim=latent_dim, image_size=img_size)
