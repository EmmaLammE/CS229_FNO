import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_wavefield(wavefield):
    ''' Plot wavefield

    Parameters
    ----------
    wavefield : 3D numpy array
        Wavefield to plot (nt, nx, ny)
        nt = number of time steps
        nx = number of grid points in x-direction
        ny = number of grid points in y-direction
    '''

    fig = plt.figure()
    ims = []

    for i in range(wavefield.shape[0]):
        caxis = wavefield.max() * 0.2
        im = plt.imshow(wavefield[i].T, vmin = -caxis, vmax=caxis, aspect = 1, cmap='seismic', animated=False)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani



def plot_model(wavefield):
    ''' Plot wavefield

    Parameters
    ----------
    wavefield : 3D numpy array
        Wavefield to plot (nt, nx, ny)
        nt = number of time steps
        nx = number of grid points in x-direction
        ny = number of grid points in y-direction
    '''

    fig = plt.figure()
    ims = []

    for i in range(wavefield.shape[0]):
        caxis = wavefield.max() * 0.2
        im = plt.imshow(wavefield[i].T, aspect = 1, cmap='jet', animated=False)
        ims.append([im])
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani


def plot_data(data1, data2, title1='Data1', title2='Data2', 
              title3='Difference', clip=99.9, ratio=100.0, aspect='auto',cmap='gray'):
    """ Plot two data
    """
    
    if not isinstance(data1, np.ndarray):
        try:
            data1 = data1.cpu().detach().numpy()
        except AttributeError:
            pass    
        
    if not isinstance(data2, np.ndarray):
        try:
            data2 = data2.cpu().detach().numpy()
        except AttributeError:
            pass    
        


    plt.figure(figsize=(10,10))
    vmax = np.percentile(data1, clip)
    
    plt.subplot(3,1,1)
    plt.imshow(data1.T, aspect=aspect, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.grid()
    plt.title(title1)
    
    plt.subplot(3,1,2)
    plt.imshow(data2.T, aspect=aspect, cmap=cmap, vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.grid()
    plt.title(title2)
    
    plt.subplot(3,1,3)
    plt.imshow(data1.T-data2.T, aspect=aspect, cmap=cmap, vmin=-vmax / ratio, vmax=vmax /ratio)
    plt.colorbar()
    plt.grid()
    plt.title(title3 + f' (X{ratio})')

    plt.tight_layout()