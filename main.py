import os
import json
import time
import math
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from utils.network import QuasittudeNet
from utils.dataset import ComplexUrbanDataset


# GLOBAL SETTINGS
plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

seed = 17
np.random.seed(seed)
torch.manual_seed(seed)
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


def train_epoch(epoch, model, loader, optimizer):
    model.train()
    train_loss = 0.
    with tqdm(total=len(loader), desc=f'Training epoch {epoch}', leave=False, colour='green') as pbar:
        for _, sensor_data, _, rotated_g in loader:
            sensor_data = sensor_data.to(device)
            rotated_g = rotated_g.to(device)
            decoded, rp_latent = model(sensor_data)
            loss = model.loss(sensor_data, decoded, rp_latent, rotated_g)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.update(1)
    return train_loss / len(loader.dataset)


def valid_epoch(epoch, model, loader):
    model.eval()
    valid_loss = 0.
    with torch.no_grad(), tqdm(total=len(loader), desc=f'Validating epoch {epoch}', leave=False, colour='yellow') as pbar:
        for _, sensor_data, _, rotated_g in loader:
            sensor_data = sensor_data.to(device)
            rotated_g = rotated_g.to(device)
            decoded, rp_latent = model(sensor_data)
            loss = model.loss(sensor_data, decoded, rp_latent, rotated_g)
            valid_loss += loss.item()
            pbar.update(1)
    return valid_loss / len(loader.dataset)


def test_epoch(epoch, model, loader):
    model.eval()
    test_loss = 0.
    with torch.no_grad(), tqdm(total=len(loader), desc=f'Testing epoch {epoch}', leave=False, colour='blue') as pbar:
        for _, sensor_data, _ in loader:
            sensor_data = sensor_data.to(device)
            y, rp = model(sensor_data)
            loss = model.loss(y, rp)
            test_loss += loss.item()
            pbar.update(1)
    return test_loss / len(loader.dataset)


def preview(epoch, model, loader, save=False, path=None):
    model.eval()
    model_latent_sols = []
    ref_sols = []
    timestamps = []
    inferences = []
    with torch.no_grad(), tqdm(total=len(loader), desc=f'Previewing epoch {epoch}', leave=False, colour='red') as pbar:
        for t, sensor_data, ref, *_ in loader:
            tic = time.time()
            sensor_data = sensor_data.to(device)
            _, rp_latent = model(sensor_data)
            toc = time.time()
            inferences.append(toc-tic)
            timestamps.extend(t.squeeze(-1).tolist())
            ref_ = [list(entry) for entry in ref.tolist()]
            ref_sols.extend(ref_)
            rp_ = [list(entry) for entry in rp_latent.cpu().tolist()]
            model_latent_sols.extend(rp_)
            pbar.update(1)
    ref_sols = np.array(ref_sols)
    model_latent_sols = np.rad2deg(model_latent_sols)
    timestamps = np.array(timestamps).reshape(-1, 1)

    fig, ((ax1), (ax2)) = plt.subplots(2, 1, figsize=(12, 8))
    canvas = FigureCanvasAgg(fig)
    ax1.plot(model_latent_sols[:, 0], 'r', label='Latent')
    ax1.plot(ref_sols[:, 0], 'k', linewidth=0.5, label='Reference')
    ax1.set_xlim([0, model_latent_sols.shape[0]])
    ax1.set_ylabel('Roll [deg]')
    ax1.legend()

    ax2.plot(model_latent_sols[:, 1], 'r', label='Latent')
    ax2.plot(ref_sols[:, 1], 'k', linewidth=0.5, label='Reference')
    ax2.set_xlim([0, model_latent_sols.shape[0]])
    ax2.set_ylabel('Pitch [deg]')
    ax2.set_xlabel('Samples')

    plt.tight_layout()
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape((height, width, 4))
    plt.close(fig)    
    if save:
        inferences = np.array(inferences, dtype=np.float32).reshape((-1,1))
        df = pd.DataFrame(np.hstack((timestamps, ref_sols, model_latent_sols, inferences)), 
                          columns=['time','ref.roll','ref.pitch','ref.yaw','model.roll','model.pitch','inference_time'])
        df.to_csv(os.path.join(path, 'processed.csv'), index=False)
    return image_array

    
def train():
    # Defining the model
    seq_len = 100
    encoder_params ={
    'in_channels': 6,
    'channels': list(range(6, 120, 6))[:4],
    'kernel_size': 7,
    'dilation': 3,
    'stride': 1,
    'padding': 0,
    'groups': 6
    }

    decoder_params ={
        'in_channels': encoder_params['channels'][-1],
        'channels': encoder_params['channels'][::-1],
        'kernel_size': encoder_params['kernel_size'],
        'dilation': encoder_params['dilation'],
        'stride': encoder_params['stride'],
        'padding': encoder_params['padding'],
        'groups': encoder_params['groups'],
    }

    estimator_params = {
        'input_size': (1, 6, seq_len),
        'channels':[2]  # 
        }

    model = QuasittudeNet(encoder_params, decoder_params, estimator_params, [0.016, 0.011])
    stats = summary(model, input_data=torch.randn(1, 6, seq_len), device=device, depth=3, verbose=1)
    model.to(device)
    model.device = device

    # Preparing logging system
    root = os.path.join('.')
    tag = 'DAE_PINN_' + datetime.now().strftime("%d_%b_%Y_%H_%M_%S")
    writer = SummaryWriter(os.path.join(root, 'runs', tag))

    model_summary = repr(stats).replace(' ', '&nbsp;').replace( '\n', '<br/>')
    writer.add_text("Model", model_summary)
    # writer.add_graph(model, torch.randn(1, denoiser_params['in_channels'], seq_len, dtype=dtype, device=device), use_strict_trace=True)
    writer.add_text("Encoder Parameters", str(encoder_params))
    writer.add_text("Decoder Parameters", str(decoder_params))
    writer.add_text("Estimator Parameters", str(estimator_params))
    checkpoints_path = os.path.join(root, 'checkpoints', tag)
    os.makedirs(checkpoints_path, exist_ok=True)

    # Building datasets
    print('\nBuilding datasets:')

    # Getting sequences according to the split
    with open(os.path.join(root, 'data', 'complexurban', 'sets.json')) as json_file:
        parsed = json.load(json_file)
    
    params = {"split": "train", "sequences": parsed["train"], "seq_len": seq_len, "gyroscopes": True, "wheel_odometer": True}
    train_data = ComplexUrbanDataset(os.path.join(root, 'data', 'complexurban'), params)

    params = {"split": "valid", "sequences": parsed["valid"], "seq_len": seq_len, "gyroscopes": True, "wheel_odometer": True}
    valid_data = ComplexUrbanDataset(os.path.join(root,'data', 'complexurban'), params)

    params = {"split": "test", "sequences": parsed["test"], "seq_len": seq_len, "gyroscopes": True, "wheel_odometer": True}
    test_data = ComplexUrbanDataset(os.path.join(root,'data', 'complexurban'), params)
    
    # Building dataloaders
    batch_size = 1024
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True,  worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True,  worker_init_fn=seed_worker, generator=g)
    # test_loader  = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True,  worker_init_fn=seed_worker, generator=g)
    preview_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=24, pin_memory=True)
    
    # Organize parameters into groups
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': 1e-3},
        {'params': model.decoder.parameters(), 'lr': 1e-3},
        {'params': model.estimator.parameters(), 'lr': 1e-3},
        {'params': [model.misalignment], 'lr': 1e-5}
    ]
    # Preparing optimizer and schedulers
    optimizer = optim.SGD(param_groups, weight_decay=1e-5, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50)

    # Training model
    print('\nStarting training routine:')
    epochs = 399
    best_valid_loss = math.inf
    for epoch in range(epochs+1):
        # Training, validating and testing model
        tic = time.time()
        train_loss = train_epoch(epoch, model, train_loader, optimizer)
        valid_loss = valid_epoch(epoch, model, valid_loader)
        # test_loss  = test_epoch(epoch, model, test_loader)
        scheduler.step(valid_loss)
        toc = time.time()

        # Logging data
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/valid", valid_loss, epoch)
        # writer.add_scalar("Loss/test",  test_loss,  epoch)
        writer.add_scalar("Misalignment/roll", model.misalignment.detach().cpu().squeeze()[0], epoch)
        writer.add_scalar("Misalignment/pitch", model.misalignment.detach().cpu().squeeze()[1], epoch)

        if epoch == 0:
            print('-' * 75)
            print('{:^7s}{:^15s}{:^17s}{:^14s}{:^10s}{:^12s}'.format('Epoch','Training Loss','Validation Loss','Testing Loss','LR','Time [min]'))
            print('-' * 75)
        print('{:^7d}{:^15.6f}{:^17.6f}{:^14.6f}{:10.6f}{:^12.2f}'.format(epoch, train_loss, valid_loss, 0., optimizer.state_dict()['param_groups'][0]['lr'], (toc-tic)/60))

        # Saving weights
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch':epoch,
                'valid_loss':valid_loss,
                'lr':optimizer.state_dict()['param_groups'][0]['lr'],
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict()
            }, os.path.join(root, 'checkpoints', tag, f'weights_{epoch}.pt'))

        # Previewing test data
        # if epoch % 10 == 0:
        image_array = preview(epoch, model, preview_loader)
        writer.add_image('Testing Set', image_array[:,:,:3], epoch, dataformats='HWC')

        # Early stop
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            print("Learning rate below threshold, stopping training.")
            break

    writer.flush()
    writer.close()


def test():
    seq_len = 100
    encoder_params ={
    'in_channels': 6,
    'channels': list(range(6, 120, 6))[:4],
    'kernel_size': 7,
    'dilation': 3,
    'stride': 1,
    'padding': 0,
    'groups': 6
    }

    decoder_params ={
        'in_channels': encoder_params['channels'][-1],
        'channels': encoder_params['channels'][::-1],
        'kernel_size': encoder_params['kernel_size'],
        'dilation': encoder_params['dilation'],
        'stride': encoder_params['stride'],
        'padding': encoder_params['padding'],
        'groups': encoder_params['groups'],
    }

    estimator_params = {
        'input_size': (1, 6, seq_len),
        'channels':[2]  # 
        }
    model = QuasittudeNet(encoder_params, decoder_params, estimator_params, [0.016, 0.011])
    model.to(device)
    model.device = device

    # Loading pretrained weights
    root = os.path.join('.')
    tag = 'DAE_PINN_01_Apr_2024_17_06_37'
    weights = sorted(os.listdir(os.path.join(root, 'checkpoints', tag)))
    checkpoints_path = os.path.join(root, 'checkpoints', tag, weights[-1])
    weights = torch.load(checkpoints_path)
    model.load_state_dict(weights['model_state_dict'])

    # Saving results
    params = {"split": "test", "sequences": ["urban08"], "seq_len": seq_len, "gyroscopes": True, "wheel_odometer": True}
    test_data = ComplexUrbanDataset(os.path.join(root,'data', 'complexurban'), params)
    preview_loader = DataLoader(test_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=4)
    path = os.path.join(root, 'checkpoints', tag, 'sequences', params['sequences'][-1]) 
    os.makedirs(path, exist_ok=True)
    _ = preview(9999, model, preview_loader, save=True, path=path)


if __name__ == "__main__":
    # train()
    test()