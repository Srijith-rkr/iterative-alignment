import re
import pandas as pd
import matplotlib.pyplot as plt
from gradien_plots import pytorch_implementation
import numpy as np

# Path to your file
file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/NoneConfig_stage1_800_spin_256_SPIN_iter0_L40_1e-5__725725.out'
file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_800_SPIN_iter1_L40_1e-5__728816.out'

only_lora_file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_1024_lora_only_SPIN_iter0_L40_1e-5__732983.out'
only_lora_file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_1024_lora_only_SPIN_iter1_L40_1e-5__732984.out'
beta_lora_file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/beta_1e-2_stage1_1024_lora_only_SPIN_iter1_L40_1e-5__732985.out'
beta_lora_file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/beta_1e-2_stage1_1024_lora_only_SPIN_iter0_L40_1e-5__732986.out'
pattern = r"(policy_real_logps:\s*tensor\([^\)]*?\))\n(policy_generated_logps:\s*tensor\([^\)]*?\))\n(opponent_real_logps:\s*tensor\([^\)]*?\))\n(opponent_generated_logps:\s*tensor\([^\)]*?\))"


def gradient_filed_tracking(file_path,tag, color):
    data = []
    with open(file_path, 'r') as f:
        content = f.read()
        blocks = re.findall(pattern, content)

    for i in blocks:

        data.append({
            'policy_real_logp': float(i[0].split('tensor([')[1].split(']')[0]),
            'policy_generated_logp': float(i[1].split('tensor([')[1].split(']')[0]),
            'opponent_real_logp': float(i[2].split('tensor([')[1].split(']')[0]),
            'opponent_generated_logp': float(i[3].split('tensor([')[1].split(']')[0]),
            'device': int(i[0].split('device=\'cuda:')[-1].split("'")[0])
        })
        

    df = pd.DataFrame(data)
    df['X'] = df['policy_real_logp'] - df['opponent_real_logp']
    df['Y'] = df['policy_generated_logp'] - df['opponent_generated_logp']


    x_lower, x_upper = np.percentile(df['X'], [5, 95])
    df_filtered = df[(df['X'] >= x_lower) & (df['X'] <= x_upper)]
    y_lower, y_upper = np.percentile(df['Y'], [5, 95])
    df_filtered = df_filtered[(df_filtered['Y'] >= y_lower) & (df_filtered['Y'] <= y_upper)]

    step = 2000
    plot_data = []
    for start in range(0, df_filtered.shape[0],step):
        x = df_filtered['X'][start:start+step].mean()
        y = df_filtered['Y'][start:start+step].mean()
        plot_data.append({'X': x, 'Y': y})
        
    df_plot = pd.DataFrame(plot_data)
    plt.plot(df_plot['X'],df_plot['Y'] , marker='o', color=color, linestyle='-',label=f'Average location of the datapoints across iteration {tag}') # 'o' for markers, '-' for lines
    for i, (x, y) in enumerate(zip(df_plot['X'], df_plot['Y'])):
        if i % 4 == 0:
            plt.text(x, y, str(i), fontsize=9, ha='right', va='bottom')
    return df_plot


def plot_with_beta(beta, color, headwidth, width):
    winner_grid = np.zeros((10,10))
    loser_grid = np.zeros((10,10))
    for i in range(winner_grid.shape[0]):
        for j in range(winner_grid.shape[1]):
            winner_grid[i,j], loser_grid[i,j] = pytorch_implementation(winner_scale[i],loser_scale[j],1.,beta,'spin_deravative_of_log')
            # SRIJITH : Since we do gradient descent, we multiply by -1
            winner_grid[i,j] = winner_grid[i,j] * -1
            loser_grid[i,j] = loser_grid[i,j] * -1
            

    plt.quiver(winner_scale,loser_scale, winner_grid, loser_grid, angles='xy',pivot='tail',headwidth=headwidth,headlength=2,width= width ,color=color, label=f'Gradient field with Beta = {beta}')#, scale_units='xy', scale=1)
    
    
plt.figure(figsize=(10, 6))  

# plot_0 = gradient_filed_tracking(file_path_0,'iter0', 'b')
# plot_1 = gradient_filed_tracking(file_path_1,'iter1', 'r')
lora_plot_0 = gradient_filed_tracking(only_lora_file_path_0,'LORAiter0', 'g')
# lora_plot_1 = gradient_filed_tracking(only_lora_file_path_1,'LORAiter1', 'y')
# beta_lora_0 = gradient_filed_tracking(beta_lora_file_path_0,'BetaLORAiter0', 'c')
# beta_lora_1 = gradient_filed_tracking(beta_lora_file_path_1,'BetaLORAiter1', 'm')


plots = [ lora_plot_0]

# Initialize x_min, x_max, y_min, y_max with extreme values
x_min, x_max = float('inf'), float('-inf')
y_min, y_max = float('inf'), float('-inf')

# Loop through each plot and update the min and max values
for plot in plots:
    x_min = min(x_min, plot['X'].min())
    x_max = max(x_max, plot['X'].max())
    y_min = min(y_min, plot['Y'].min())
    y_max = max(y_max, plot['Y'].max())

winner_scale = np.linspace(x_min, x_max, 10)
loser_scale = np.linspace(y_min, y_max, 10)

plot_with_beta(0.01, 'r',3, 0.004 )
plot_with_beta(0.1, 'y',4, 0.003 )    
plot_with_beta(1., 'b',2, 0.005 )
# plot_with_beta(0.01, 'b',2, 0.005 )

# Add title and labels
plt.title('Location of datapoints on gradient field across iterations 0')
plt.xlabel(r'$log\left( \frac{\pi_{pol.}^{Winner}}{\pi_{ref.}^{Winner}} \right)$' ,fontsize=12)
plt.ylabel(r'$log\left( \frac{\pi_{pol.}^{Loser}}{\pi_{ref.}^{Loser}} \right)$',  fontsize=12)
plt.legend(loc='upper left')
# plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.grid()
# Show the plot
plt.savefig('/home/srijithr/iterative-alignment/gradient_field/gradient_filed.jpeg', dpi= 500)
