
import os
import torch
import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset
from tqdm import tqdm
from math import pi


class ComplexUrbanDataset(Dataset):

    def __init__(self, path, params) -> None:
        super(ComplexUrbanDataset, self).__init__()
        self.path = path
        self.params = params 
        self.data = {}
        self.indices = []

        # Reading data for each selected sequence
        pbar = tqdm(total=len(params['sequences']), desc=f"Creating {params['split']} set")
        for sequence in params['sequences']:
            sequence_path = os.path.join(self.path, sequence)
            # Reading reference
            ref_data = pd.read_csv(os.path.join(sequence_path, 'global_pose.csv'), float_precision='round_trip', header=None)
            ref_time = ref_data.iloc[:, 0].to_numpy() * 10**-9
            ref_R = ref_data.iloc[:, [1,2,3,5,6,7,9,10,11]].to_numpy()
            ref_R = ref_R[np.newaxis, ...].reshape(-1, 3, 3, order='C')
            ref_euler = R.from_matrix(ref_R).as_euler('xyz', degrees=True)

            # Reading wheel odometer
            wheel_data = pd.read_csv(os.path.join(sequence_path, 'sensor_data', 'encoder.csv'), float_precision='round_trip', header=None)
            wheel_time = wheel_data.iloc[:, 0].to_numpy() * 10**-9
            wheel_left_count = wheel_data.iloc[:, 1].to_numpy()
            wheel_right_count = wheel_data.iloc[:, 2].to_numpy()
            with open(os.path.join(sequence_path, 'calibration', 'EncoderParameter.txt'), 'r') as f:
                f_lines = f.readlines()
                encoder_resolution = int(f_lines[1].strip('\n ').split(':')[-1])
                left_wheel_diameter = float(f_lines[2].strip('\n ').split(':')[-1])
                right_wheel_diameter = float(f_lines[3].strip('\n ').split(':')[-1])
            wheel_time, wheel_speed = self._get_average_speed(wheel_time, wheel_left_count, wheel_right_count, encoder_resolution, left_wheel_diameter, right_wheel_diameter)
            wheel_diff_time = np.diff(wheel_time)
            wheel_acceleration = np.diff(wheel_speed) / wheel_diff_time
            wheel_acceleration = np.concatenate(([0.], wheel_acceleration))

            # Reading imu
            imu_data = pd.read_csv(os.path.join(sequence_path, 'sensor_data', 'xsens_imu.csv'), float_precision='round_trip', header=None)
            imu_time = imu_data.iloc[:, 0].to_numpy() * 10**-9
            imu_gyro_accel_data = imu_data.iloc[:, [8, 9, 11, 12, 13]].to_numpy()  # 10

            # Interpolate data based on a inclusive timeline
            step = 1 / 100
            timeline_start = max(ref_time.min(), wheel_time.min(), imu_time.min())
            timeline_end   = min(ref_time.max(), wheel_time.max(), imu_time.max())
            timeline = np.arange(timeline_start, timeline_end, step)

            ref_euler_x = self._interpolate_angles(ref_time, timeline, ref_euler[:, 0])
            ref_euler_y = self._interpolate_angles(ref_time, timeline, ref_euler[:, 1])
            ref_euler_z = self._interpolate_angles(ref_time, timeline, ref_euler[:, 2])
            ref_euler = np.concatenate((ref_euler_x.reshape(-1,1), ref_euler_y.reshape(-1,1), ref_euler_z.reshape(-1,1)), axis=1)
            
            wheel_speed = np.interp(timeline, wheel_time, wheel_speed)
            # wheel_acceleration = np.interp(timeline, wheel_time, wheel_acceleration)

            imu_gyro_accel_data_wx = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 0])
            imu_gyro_accel_data_wy = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 1])
            # imu_gyro_accel_data_wz = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 2])
            imu_gyro_accel_data_fx = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 2])
            imu_gyro_accel_data_fy = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 3])
            imu_gyro_accel_data_fz = np.interp(timeline, imu_time, imu_gyro_accel_data[:, 4])
            imu_gyro_accel_data = np.concatenate((imu_gyro_accel_data_wx.reshape(-1,1), imu_gyro_accel_data_wy.reshape(-1,1), #imu_gyro_accel_data_wz.reshape(-1,1),
                                                  imu_gyro_accel_data_fx.reshape(-1,1), imu_gyro_accel_data_fy.reshape(-1,1), imu_gyro_accel_data_fz.reshape(-1,1)), axis=1)
            
            sensor_data = np.concatenate((imu_gyro_accel_data, wheel_speed.reshape(-1,1)), axis=1)
            
            # Storing data
            self.data[sequence] = [timeline, sensor_data, ref_euler]
            sequence_indices = np.arange(params['seq_len'], timeline.shape[0], 1)
            sequence_indices = [[sequence, i] for i in sequence_indices]
            self.indices.extend(sequence_indices)
            pbar.update(1)

        self.indices = tuple(self.indices)
        pbar.close()
        print(f"Initialization of the {params['split']} set complete. A total of {len(self.indices)} synchronized frames available.")

    def _get_average_speed(self, timestamps, left_count, right_count, encoder_resolution, left_wheel_diameter, right_wheel_diameter):
        time_intervals = np.diff(timestamps)
        timestamps  = timestamps[1:]
        left_count  = np.diff(left_count)
        right_count = np.diff(right_count)        
        distance_per_count_left = pi * left_wheel_diameter / encoder_resolution
        distance_per_count_right = pi * right_wheel_diameter / encoder_resolution
        distances_left = left_count * distance_per_count_left
        distances_right = right_count * distance_per_count_right
        average_distances = (distances_left + distances_right) / 2
        average_speeds = average_distances / time_intervals
        return timestamps, average_speeds
    
    def _interpolate_angles(self, old_timeline, new_timeline, signal):
        sa = np.sin(np.deg2rad(signal))
        ca = np.cos(np.deg2rad(signal))
        sa_smoothed = np.interp(new_timeline, old_timeline, sa) 
        ca_smoothed = np.interp(new_timeline, old_timeline, ca)
        new_signal = np.rad2deg(np.arctan2(sa_smoothed, ca_smoothed))
        return new_signal
    
    def _accelerometers_noise(self):
        min_noise_density = 5.886e-3
        # max_noise_density = 5.886e-3
        # noise_density = np.random.uniform(min_noise_density, max_noise_density)
        return np.random.normal(0., min_noise_density, (self.params['seq_len'], 3))
    
    def _gyroscopes_noise(self):
        min_noise_density = 1.745e-4
        # max_noise_density = 1.745e-3
        # noise_density = np.random.uniform(min_noise_density, max_noise_density)
        return np.random.normal(0., min_noise_density, (self.params['seq_len'], 2))
    
    def _wheel_odometer_noise(self):
        min_noise_density = 0.05 
        # max_noise_density = 0.10
        # noise_density = np.random.uniform(min_noise_density, max_noise_density)
        return np.random.normal(0., min_noise_density, (self.params['seq_len'], 1))
    
    def _accelerometers_bias(self):
        min_bias_in_run_instability = 1.4715e-4
        # max_bias_in_run_instability = 1.4715e-3
        # bias_in_run_instability = np.random.uniform(min_bias_in_run_instability, max_bias_in_run_instability)
        return np.random.normal(0., min_bias_in_run_instability, (self.params['seq_len'], 3))
    
    def _gyroscopes_bias(self):
        min_bias_in_run_instability = 4.85e-5
        # max_bias_in_run_instability = 4.85e-4
        # bias_in_run_instability = np.random.uniform(min_bias_in_run_instability, max_bias_in_run_instability)
        return np.random.normal(0., min_bias_in_run_instability, (self.params['seq_len'], 2))

    def _rotated_g(self):
        angles = np.random.normal(0., 5.0, 2)
        r_angles = R.from_euler('xy', angles, degrees=True).as_matrix()
        g = np.array([0., 0., 9.80665])
        rotated_g = r_angles.T.dot(g)
        return rotated_g
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        sequence, final_index = self.indices[index]
        timeline, sensor_data, ref_euler = self.data[sequence]
        t = timeline[final_index-1]
        r = ref_euler[final_index-1, :]
        sensor_data = sensor_data[final_index-self.params['seq_len']:final_index]
        # Getting sensor data
        index_acc = [2, 3, 4]
        index_gyr = []
        index_whl = []
        if self.params['gyroscopes']:
            index_gyr = [0, 1]
        if self.params['wheel_odometer']:
            index_whl = [5]
        sensor_data = sensor_data[:, sorted(index_acc + index_gyr + index_whl)]
        # Augmenting data in case of training set
        # Accelerometers
        accelerometers_noise = self._accelerometers_noise() if self.params['split'] == 'train' else np.zeros((self.params['seq_len'], 3))
        accelerometers_bias = self._accelerometers_bias() if self.params['split'] == 'train' else np.zeros((self.params['seq_len'], 3))
        rotated_g = self._rotated_g() if self.params['split'] == 'train' else np.zeros(3)
        rotated_g_tile = np.tile(rotated_g, (self.params['seq_len'], 1)) 
        augmentation = accelerometers_noise + accelerometers_bias + rotated_g_tile
        # Gyroscopes
        if self.params['gyroscopes']:
            gyroscopes_noise = self._gyroscopes_noise() if self.params['split'] == 'train' else np.zeros((self.params['seq_len'], 2))
            gyroscopes_bias = self._gyroscopes_bias() if self.params['split'] == 'train' else np.zeros((self.params['seq_len'], 2))
            augmentation = np.hstack((gyroscopes_noise + gyroscopes_bias, augmentation))
        # Wheel Odometer
        if self.params['wheel_odometer']:
            wheel_odometer_noise = self._wheel_odometer_noise() if self.params['split'] == 'train' else np.zeros((self.params['seq_len'], 1))
            augmentation = np.hstack((augmentation, wheel_odometer_noise))
        sensor_data = sensor_data + augmentation
        rotated_g = torch.tensor(rotated_g, dtype=torch.float) / 9.80665
        # Normalizing data
        sensor_data /= 9.80665
        # Converting data type
        t = torch.tensor([t], dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        sensor_data = torch.tensor(sensor_data, dtype=torch.float)
        # Channels first
        sensor_data = sensor_data.T
        return t, sensor_data, r, rotated_g