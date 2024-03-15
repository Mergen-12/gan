import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
latent_dim = 100
image_size = 64
channels = 3
batch_size = 64
num_epochs = 100
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999

# Load data
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channels)], [0.5 for _ in range(channels)])
])

dataset = datasets.ImageFolder('C:\\Users\\dvt\\Documents\\Dataset\\img_align_celeba_folder', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise):
        return self.main(noise)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.main(img)

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Loss functions
criterion = nn.BCELoss()

# Optimizers
gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta_1, beta_2))
disc_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta_1, beta_2))

# Training loop
for epoch in range(num_epochs):
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)

        # Train Discriminator
        disc_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(noise)
        disc_real = discriminator(real_imgs).view(-1)
        disc_fake = discriminator(fake_imgs).view(-1)
        disc_loss = criterion(disc_real, torch.ones_like(disc_real, device=device)) + \
                    criterion(disc_fake, torch.zeros_like(disc_fake, device=device))
        disc_loss.backward()
        disc_optimizer.step()

        # Train Generator
        gen_optimizer.zero_grad()
        noise = torch.randn(batch_size, latent_dim, 1, 1).to(device)
        fake_imgs = generator(noise)
        gen_loss = criterion(discriminator(fake_imgs).view(-1), torch.ones_like(disc_real, device=device))
        gen_loss.backward()
        gen_optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] Discriminator Loss: {disc_loss.item():.4f}, Generator Loss: {gen_loss.item():.4f}")

# Save the trained models
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')