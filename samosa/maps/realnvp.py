"""
Class file for the RealNVP transport map using "https://github.com/xqding/RealNVP"
This repo is used for density estimation that is exactly what I need
"""

# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

from samosa.core.map import TransportMap
from samosa.core.state import ChainState
from scipy.stats import multivariate_normal
from samosa.utils.post_processing import get_position_from_states
from typing import List

class RealNVPMap(TransportMap):
    """
    Class for the RealNVP transport map using pytorch.
    """

    def __init__(self, masks, hidden_dim, samples, learning_rate):
        # realNVP = RealNVP_2D(masks, hidden_dim)
        # if torch.cuda.device_count():
        #     realNVP = realNVP.cuda()
        # device = next(realNVP.parameters()).device

        X_tensor = torch.Tensor(samples)

        ## Create dataset and dataloader (keep same structure)
        batch_size = 500
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        ## Initialize model on CPU
        device = torch.device("cpu")
        realNVP = RealNVP_2D(masks, hidden_dim).to(device)
        optimizer = optim.Adam(realNVP.parameters(), lr=learning_rate)

        ## Training loop remains similar
        num_epochs = 100
        print_interval = 10

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                X = batch[0].to(device)
                
                # Forward pass through model
                z, logdet = realNVP.inverse(X)
                
                # Loss calculation (same as before)
                loss = torch.log(z.new_tensor([2*math.pi])) + torch.mean(torch.sum(0.5*z**2, -1) - logdet)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Print training progress
            avg_loss = epoch_loss / num_batches
            if (epoch + 1) % print_interval == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.5f}")

        self.realNVP = realNVP    


class Affine_Coupling(nn.Module):
    def __init__(self, mask, hidden_dim):
        super(Affine_Coupling, self).__init__()
        self.input_dim = len(mask)
        self.hidden_dim = hidden_dim

        ## mask to seperate positions that do not change and positions that change.
        ## mask[i] = 1 means the ith position does not change.
        self.mask = nn.Parameter(mask, requires_grad = False)

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.input_dim)
        self.scale = nn.Parameter(torch.Tensor(self.input_dim))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.input_dim)

    def _compute_scale(self, x):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(x*self.mask))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale        
        return s

    def _compute_translation(self, x):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.translation_fc1(x*self.mask))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)        
        return t
    
    def forward(self, x):
        ## convert latent space variable to observed variable
        s = self._compute_scale(x)
        t = self._compute_translation(x)
        
        y = self.mask*x + (1-self.mask)*(x*torch.exp(s) + t)        
        logdet = torch.sum((1 - self.mask)*s, -1)
        
        return y, logdet

    def inverse(self, y):
        ## convert observed varible to latent space variable
        s = self._compute_scale(y)
        t = self._compute_translation(y)
                
        x = self.mask*y + (1-self.mask)*((y - t)*torch.exp(-s))
        logdet = torch.sum((1 - self.mask)*(-s), -1)
        
        return x, logdet
    
class RealNVP_2D(nn.Module):
    '''
    A vanilla RealNVP class for modeling 2 dimensional distributions
    '''
    
    def __init__(self, masks, hidden_dim):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RealNVP_2D, self).__init__()        
        self.hidden_dim = hidden_dim        
        self.masks = nn.ParameterList(
            [nn.Parameter(torch.Tensor(m),requires_grad = False)
             for m in masks])

        self.affine_couplings = nn.ModuleList(
            [Affine_Coupling(self.masks[i], self.hidden_dim)
             for i in range(len(self.masks))])
        
    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        # # a normalization layer is added such that the observed variables is within
        # # the range of [-4, 4].
        # logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)        
        # y = 4*torch.tanh(y)
        # logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables        
        x = y        
        logdet_tot = 0

        # inverse the normalization layer
        # logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        # x  = 0.5*torch.log((1+x/4)/(1-x/4))
        # logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot
    


# Map induced pdf
def pullback_pdf(self, rho, x):
    x = x.T
    device = torch.device("cpu")
    X_test_tensor = torch.Tensor(x).to(device)
    r, logdet = self.realNVP.inverse(X_test_tensor)
    with torch.no_grad():
        r = r.cpu().numpy()
        logdet = logdet.cpu().numpy()
    log_pdf = rho(r) + logdet
    return np.exp(log_pdf)

def plot_contours(self, pdf, fname):
    
    n_points = 10000
    x = np.linspace(-3, 3, int(np.sqrt(n_points)))
    y = np.linspace(-8, 8, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate PDFs for grid points
    Z = np.exp(pdf(grid_points).reshape(X.shape))
    ref_distribution = multivariate_normal(np.zeros(2),np.eye(2))  #standard normal
    # ref_pdf_at_grid = ref_distribution.pdf(grid_points.T)
    ref = lambda x: ref_distribution.logpdf(x)

    map_induced_pdf = self.pullback_pdf(ref,grid_points.T).reshape(X.shape)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.contour(X, Y, Z, levels=5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.contour(X, Y, map_induced_pdf, levels=5, linestyles='dashed')
    plt.tight_layout()

    # Automatically adjust xlim and ylim
    # plt.xlim(X.min(), X.max())
    # plt.ylim(Y.min(), Y.max())

    plt.savefig(f'{fname}.png')
    plt.close()

def plot_contour_scatter(self, pdf, samples, fname):
    
    n_points = 10000
    x = np.linspace(-3, 3, int(np.sqrt(n_points)))
    y = np.linspace(-4, 8, int(np.sqrt(n_points)))
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack([X.ravel(), Y.ravel()])

    # Evaluate PDFs for grid points
    Z = np.exp(pdf(grid_points).reshape(X.shape))

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.contour(X, Y, Z, levels=5, linestyles='dashed')
    plt.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()

    plt.savefig(f'{fname}.png')
    plt.close()