from torch import nn
import torch.nn.functional as F




LRELU_SLOPE = 0.1


class Generator(nn.Module):
    
    def __init__(self, h=None):
        super(Generator, self).__init__()
        
        nz = h.nz
        ngf = h.ngf
        nc = h.nc
                
        self.main = nn.Sequential(
        
             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(LRELU_SLOPE),
            
             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(LRELU_SLOPE),
            
             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(LRELU_SLOPE),
            
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(LRELU_SLOPE),
            
            # state size. (ngf) 
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
        self.main.apply(init_weights)
        
    
    def forward(self, x):
        return self.main(x)
   



  class Discriminator(nn.Module):
    
    def __init__(self, h=None):
        super(Discriminator, self).__init__()
        
        ndf = h.ndf
        nc = h.nc
        
        self.main = nn.Sequential(
        
            # input is (h.nc) x 64 x 32
            
             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(LRELU_SLOPE, inplace=True),
            
              nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
              nn.LeakyReLU(LRELU_SLOPE, inplace=True) ,
            
           nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(LRELU_SLOPE, inplace=True),
           
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(LRELU_SLOPE, inplace=True),
            
             
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            
        )
     
        
        
        self.main.apply(init_weights)
        
        
        
    def forward(self, x):
        
        return self.main(x)
    
    
    


def init_weights(m):
    classname = m.__class__.__name__
    
    if classname.find("Conv") != -1:
         nn.init.normal_(m.weight.data, 0.0, 0.02)
        
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)




