import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

# Loading data from the selected trajectory
root = os.path.join('~', 'Documents', 'Github', 'quasittudenet')
sequence = os.path.join(root, 'data', 'complexurban', 'urban08')
xsens = os.path.join(sequence, 'sensor_data', 'xsens_imu.csv')
xsens_df = pd.read_csv(xsens, header=None, float_precision='round_trip')

# Extracting linear accelerations and the sensor's onboard solution
xsens_acc = xsens_df.iloc[:, [11, 12, 13]].to_numpy()
xsens_att = xsens_df.iloc[:, [5, 6, 7]].to_numpy()

# Plotting azimuth for verification of static portions
plt.plot(xsens_att[:, 2])
plt.show()

# Plotting fz (gravity) for further verification of static portions
plt.plot(xsens_acc[:, 2])
plt.show()

# Defining interval to be used for calibration
N0 = 0
N1 = 500
acc_portion = xsens_acc[N0:N1, :]

# Defining standard gravity field for alignment
gravity = np.zeros((acc_portion.shape[0], 3))
gravity[:, 2] = 9.80665

# Aligning vector to recover the Eulers angles
rot, rssd, sens = R.align_vectors(acc_portion, gravity, return_sensitivity=True)
print(rot.as_euler('xyz', degrees=False).tolist())