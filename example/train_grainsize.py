import sys
sys.path.append('../src/')

import argparse
from timeit import default_timer

import torch
import numpy as np

from model import FNO2d
from utilities3 import count_params, LpLoss


def load_data(strainrate_path, temperature_path, grainsize_path, T_in, T_end, 
              batch_size = 2, shuffle=True):
    
    """ Load the data from the given paths and split into train, validation and test sets

    Parameters
    ----------
    vp_path : str 
        Path to the vp data in .npy format with shape (Nsmaple, nx, nz)
    vs_path : str
        Path to the vs data in .npy format (Nsmaple, nx, nz)
    rho_path : str
        Path to the rho data in .npy format (Nsmaple, nx, nz)
    vx_path : str
        Path to the vx data in .npy format (Nsmaple, nt, nx, nz)
    T_in : int
        Number of time steps in the input
    T_end : int
        Number of time steps in the output
    batch_size : int
        Batch size for the data loader
    shuffle : bool
        Whether to shuffle the data or not
    """

    # Load the input data
    strainrate = np.load(strainrate_path)
    temperature = np.load(temperature_path)

    # Load the output data
    grainsize = np.load(grainsize_path)

    # convert to torch tensors
    strainrate = torch.from_numpy(strainrate).float()
    temperature = torch.from_numpy(temperature).float()
    grainsize = torch.from_numpy(grainsize).float()

    # Expand the dimensions of the input data to match the output data (ns, nx, nz, 1)
    strainrate = strainrate.unsqueeze(-1)
    temperature = temperature.unsqueeze(-1)

    # permute the dimensions of the output data
    grainsize = grainsize.permute(0, 2, 3, 1)

    # split the data into training, validation and test sets in the ratio 80:10:10 randomly
    random_index = np.random.permutation(vp.shape[0])
    train_index = random_index[0:int(0.8 * vp.shape[0])]
    valid_index = random_index[int(0.8 * vp.shape[0]):int(0.9 * vp.shape[0])]
    test_index = random_index[int(0.9 * vp.shape[0]):]

    # train data
    vp_train = vp[train_index, :, :, :]
    vs_train = vs[train_index, :, :, :]
    rho_train = rho[train_index, :, :, :]
    vx_train = vx[train_index, :, :, :]

    # validation data
    vp_valid = vp[valid_index, :, :, :]
    vs_valid = vs[valid_index, :, :, :]
    rho_valid = rho[valid_index, :, :, :]
    vx_valid = vx[valid_index, :, :, :]

    # test data
    vp_test = vp[test_index, :, :, :]
    vs_test = vs[test_index, :, :, :]
    rho_test = rho[test_index, :, :, :]
    vx_test = vx[test_index, :, :, :]

    # split the time steps
    vx_train_a = vx_train[:, :, :, 0:T_in]
    vx_train_u = vx_train[:, :, :, T_in:T_end]
    vx_valid_a = vx_valid[:, :, :, 0:T_in]
    vx_valid_u = vx_valid[:, :, :, T_in:T_end]

    print("Information about the data:")
    print("Shape of vp_train   : ", vp_train.shape)
    print("Shape of vs_train   : ", vs_train.shape)
    print("Shape of rho_train  : ", rho_train.shape)
    print("Shape of vx_train_a : ", vx_train_a.shape)
    print("Shape of vx_train_u : ", vx_train_u.shape)
    print("Shape of vp_valid   : ", vp_valid.shape)
    print("Shape of vs_valid   : ", vs_valid.shape)
    print("Shape of rho_valid  : ", rho_valid.shape)
    print("Shape of vx_train_a : ", vx_valid_a.shape)
    print("Shape of vx_valid_u : ", vx_valid_u.shape)

    # # save data to disk
    # for data, path in zip([vp_train, vs_train, rho_train, vx_train], [vp_path, vs_path, rho_path, vx_path, vx_path]):
    #     torch.save(data, path.replace('.npy', '_train.pt'))

    # for data, path in zip([vp_valid, vs_valid, rho_valid, vx_valid], [vp_path, vs_path, rho_path, vx_path, vx_path]):
    #     torch.save(data,  path.replace('.npy', '_valid.pt'))

    for data, path in zip([vp_test, vs_test, rho_test, vx_test], [vp_path, vs_path, rho_path, vx_path, vx_path]):
        torch.save(data,  path.replace('.npy', '_test.pt'))

    # create the train and test loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vx_train_a, vx_train_u, vp_train, vs_train, rho_train), 
        batch_size = batch_size, shuffle=shuffle)
    
    valid_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(vx_valid_a, vx_valid_u, vp_valid, vs_valid, rho_valid),
        batch_size = batch_size, shuffle=shuffle)

    return train_loader, valid_loader


def train(model, train_loader, valid_loader, optimizer, scheduler, 
          myloss, costFunc, epoch, device, T_in, T_end, step, batch_size):

    print(f"\nThe model has {count_params(model)} trainable parameters\n")
    print(f"Training the model on {device} for {epoch} epoch ...\n")

    train_loss = torch.zeros(epoch)
    test_loss  = torch.zeros(epoch)

    for ep in range(epoch):
        model.train()
        t1 = default_timer()
        train_l2_step_training = 0
        test_l2_step_testing = 0
        for xx, yy, C1, C2, C3 in train_loader:

            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            C1 = C1.to(device)
            C2 = C2.to(device)
            C3 = C3.to(device)

            for t in range(0, (T_end - T_in),step):
                y = yy[..., t:t + step]
                im_train = model(xx,C1,C2,C3)
                loss += myloss(im_train.reshape(batch_size, -1), y.reshape(batch_size, -1))
                if t == 0:
                    pred = im_train
                else:
                    pred = torch.cat((pred, im_train), -1)

                xx = torch.cat((xx[..., step:], im_train), dim=-1)

            train_l2_step_training += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[ep] = train_l2_step_training
        
        with torch.no_grad():
            for xx, yy, C1, C2, C3 in valid_loader:
                loss = 0
                xx = xx.to(device)
                yy = yy.to(device)
                C1 = C1.to(device)
                C2 = C2.to(device)
                C3 = C3.to(device)

                for t in range(0, (T_end - T_in),step):
                    y = yy[..., t:t + step]
                    im_test = model(xx,C1,C2,C3)
                    loss += myloss(im_test.reshape(batch_size, -1), y.reshape(batch_size, -1))
                    if t == 0:
                        pred = im_test
                    else:
                        pred = torch.cat((pred, im_test), -1)

                    xx = torch.cat((xx[..., step:], im_test), dim=-1)

                test_l2_step_testing += loss.item()
                test_loss[ep] = test_l2_step_testing

        t2 = default_timer()
        #scheduler.step()
        print(f"current epoch {ep: 6d}, time {t2 -t1 :.2f}s, train loss {train_l2_step_training :.4e}, valid loss {test_l2_step_testing :.4e}")

    return train_loss, test_loss


def main(vp_path, vs_path, rho_path, vx_path, save_path):
    """ Train the model using the given data
    """

    # define the hyperparameters
    epochs          = 500         
    learning_rate   = 0.001
    scheduler_step  = 30
    scheduler_gamma = 0.1
    sub             = 1
    T_in            = 50
    T_end           = 350
    S               = 64
    batch_size      = 256
    mode1           = 12
    mode2           = 12
    width           = 15
    step            = T_end - T_in

 
    # define the device for training (only on one GPU if available)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load the data
    train_loader, valid_loader = load_data(vp_path, vs_path, rho_path, 
                                          vx_path, T_in, T_end, 
                                          batch_size, True)

    # define the model
    model = FNO2d(mode1, mode2, width).cuda()

    # define the optimizer and the scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    # define the loss function
    myloss = LpLoss(size_average=False)
    costFunc = torch.nn.MSELoss(reduction='sum')

    # train the model
    train_loss, valid_loss = train(model, train_loader, valid_loader, 
                                  optimizer, scheduler, myloss, costFunc, 
                                  epochs, device, T_in, T_end, step, batch_size)
    
    # save the model and the loss tensors
    torch.save(model.state_dict(), save_path)
    torch.save(train_loss, save_path + '_train_loss.pt')
    torch.save(valid_loss, save_path + '_valid_loss.pt') 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model')
    parser.add_argument('--vp_path', type=str, default='data/vp.npy', help='Path to the vp data')
    parser.add_argument('--vs_path', type=str, default='data/vs.npy', help='Path to the vs data')
    parser.add_argument('--rho_path', type=str, default='data/rho.npy', help='Path to the rho data')
    parser.add_argument('--vx_path', type=str, default='data/vx.npy', help='Path to the vx data')
    parser.add_argument('--save_path', type=str, default='model.pth', help='Path to save the model')
    args = parser.parse_args()

    main(args.vp_path, args.vs_path, args.rho_path, args.vx_path, args.save_path)
