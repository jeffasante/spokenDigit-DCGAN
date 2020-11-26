import os
import argparse
import json

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn


from env import AttrDict, build_env
from models import discriminator, generator
from mel_dataset import mel_spectrogram, AudioFolder







def train(h):
    # load data
    full_dataset  = AudioFolder(audio_data, h,
                          base_mels_path=numpy_save_directory,)

    data_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)


    # create the generator
    generator = Generator(h).to(device)
    print(generator)
    print()
    
    # create the discriminator
    discriminator = Discriminator(h).to(device)
    print(discriminator)
    print()
    
    # initialize BCELoss function
    criterion = nn.BCELoss()

    # create batch of latent vectors that we will use to visualize
    # the progression of the generator
    fixed_noise = torch.randn(h.height, h.nz, 1, 1, device=device)


    # establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # optimizers for both nets
    optimizerD = optim.Adam(discriminator.parameters(), lr=h.lr,
                            betas=(h.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=h.lr,
                            betas=(h.beta1, 0.999))


    
    
    
    
    # Training loop
    
    # List to keep track of progress
    mel_spec_list = []
    G_losses = []
    D_losses = []
    
    iters = 0
    
    print('Starting Training Loop...')
    
    generator.train()
    discriminator.train()
    
    # for each epoch
    for epoch in range(h.epochs):
        
        for i, (mel, fname) in enumerate(data_loader, 0):
            
            ## Updated discriminator network
            
            # update the discriminator network
            discriminator.zero_grad()
            
            real_cpu = mel.to(device)
            b_size = real_cpu.size(0)
            label = torch.full((256,), 
                               real_label, dtype=torch.float, device=device)
            
            # forward pass real batch through the discriminator
            output = discriminator(real_cpu.float()).view(-1)
            
            # calculate loss on all-real batch
            errD_real = criterion(output, label)
            
            # calculate gradients for discriminator in backwad pass
            errD_real.backward()
            D_x = output.mean().item()
            
            
            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(256, h.nz, 1, 1,device=device)
            
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            
            # classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            
            # calculate discriminator's loss on all-fake batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            
            # Add the gradients from all-real and all-ffake batches
            errD = errD_real + errD_fake
            
            # update discriminator
            optimizerD.step()
                   
            
            ## Update generator's network
            generator.zero_grad()
            label.fill_(real_label) # fake labels are real for generator cost
            
            # since discriminator was updated, perform another pass of all-fake
            # through discrimintor
            output = discriminator(fake).view(-1)
            
            # calculate generator's loss based on this output
            errG = criterion(output, torch.zeros_like(output))
            
            # calculate gradients for G
            D_G_z2 = output.mean().item()
            
            # update generator
            optimizerG.step()
            
            
            # output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f' \
                      % (epoch, epochs, i, len(data_loader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            # print('\nHere -> ', i)
            
            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 100 == 0) or ((epoch == epochs-1) and (i == len(data_loader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                mel_spec_list.append(fake.squeeze().numpy())

            iters += 1
            
    print('\nDone.....') 
    
    return G_losses, D_losses, mel_spec_list


def main():
    print('Initializing the Training Process..')
    
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument('--input_wavs_dir', default='data/recordings')
    parser.add_argument('--input_mels_dir', default='processed_spokenDigits_np')
    parser.add_argument('--config', default='processed_spokenDigits_np')
    parser.add_argument('--training_epochs', default='1000')
    
    a = parser.parse_args()
    
    with open(a.config) as f:
        data = f.read()
        
        
    json_config = json.loads(data)
    h = AttrDict(json_config)
    
    build_env(a.config, 'config.json', a.checkpoint_path)
    
    torch.manual_seed(h.seed):
    
    if torch.cuda.is_availale(h.seed):
        torch.cuda.manual_seeed(h.seed)
        
        h.batch_size = int(h.batch_size / h.num_gpu)
    else:
        print('\nRunning on cpu')
        
        
    # train now--    
        g_losses, d_losses, generated_mels = train(h) 
    
    # visualize the loss as the network trained
    plt.plot(g_losses, d_losses)
    plt.xlabel('100\'s of batches')
    plt.ylabel('loss')
    plt.grid(True)
    # plt.ylim(0, 2.5) # consistent scale
    plt.show()
    
if __name__ == '__main__':
    main()
    

