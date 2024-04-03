import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from sklearn.metrics import mean_squared_error

plt.rcParams['figure.dpi'] = 150
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 14
})


# Low-pass filters for filtering the model's output if desired
def low_pass_filter(signal, alpha):
    b = [alpha]  
    a = [1, alpha - 1] 
    filtered_signal = lfilter(b, a, signal)
    return filtered_signal

def real_time_low_pass_filter(previous_output, current_input, alpha):
    current_output = alpha * current_input + (1 - alpha) * previous_output
    return current_output


root = os.path.join('~', 'Documents', 'Github', 'quasittudenet')

# ComplexUrban Results
tag = 'DAE_PINN_01_Apr_2024_17_06_37'
path = os.path.join(root, 'checkpoints', tag, 'sequences', 'urban14','processed.csv')
df_1 = pd.read_csv(path, float_precision='round_trip')
path = os.path.join(root, 'checkpoints', tag, 'sequences', 'urban13','processed.csv')
df_2 = pd.read_csv(path, float_precision='round_trip')
path = os.path.join(root, 'checkpoints', tag, 'sequences', 'urban08','processed.csv')
df_3 = pd.read_csv(path, float_precision='round_trip')

# Filter each sequence individually, i.e., df_1, df_2, and df_3 
# filtered_roll = [df.loc[0, 'model.roll']]
# filtered_pitch = [df.loc[0, 'model.pitch']]
# for i in range(1, df.shape[0]):
#     filtered_roll.append(real_time_low_pass_filter(filtered_roll[-1], df.loc[i, 'model.roll'],  0.2))
#     filtered_pitch.append(real_time_low_pass_filter(filtered_pitch[-1], df.loc[i, 'model.pitch'], 0.2))
# df['filtered.roll'] = filtered_roll  #low_pass_filter(df.loc[:, 'model.roll'].to_numpy(), 0.05)
# df['filtered.pitch'] = filtered_pitch  #low_pass_filter(df.loc[:, 'model.pitch'].to_numpy(), 0.05)

# Plotting results
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(df_1.loc[:, 'model.roll'], 'b')
# ax1.plot(df.loc[:, 'filtered.roll'], 'r')
ax1.plot(df_1.loc[:, 'ref.roll'], color='#FF9505', linestyle='-')
ax1.set_xlim([0, df_1.shape[0]])
ax1.set_ylabel('$\phi [{}^\circ]$')
ax1.grid(linestyle=':')

ax2.plot(df_1.loc[:, 'model.pitch'], 'b')
# ax2.plot(df.loc[:, 'filtered.pitch'], 'r')
ax2.plot(df_1.loc[:, 'ref.pitch'], color='#FF9505', linestyle='-')
ax2.set_xlim([0, df_1.shape[0]])
ax2.set_ylabel('$\\theta [{}^\circ]$')
ax2.set_xlabel('Samples')
ax2.grid(linestyle=':')

plt.tight_layout()
plt.show()

# Computing RMSE over the testing set
df = pd.concat([df_1, df_2, df_3])
print(f"Roll RMSE: {mean_squared_error(df.loc[:, 'ref.roll'].to_numpy(), df.loc[:, 'model.roll'].to_numpy(), squared=False)}")
print(f"Pitch RMSE: {mean_squared_error(df.loc[:, 'ref.pitch'].to_numpy(), df.loc[:, 'model.pitch'].to_numpy(), squared=False)}")

# Roll misalignment
print(f"Roll Misalignment: {np.mean(df.loc[:, 'ref.roll'].to_numpy() - df.loc[:, 'model.roll'].to_numpy())}")