import re
import pandas as pd
import matplotlib.pyplot as plt
from gradien_plots import pytorch_implementation
import numpy as np

# Path to your file


pattern = r"(policy_real_logps:\s*tensor\([^\)]*?\))\n(policy_generated_logps:\s*tensor\([^\)]*?\))\n(opponent_real_logps:\s*tensor\([^\)]*?\))\n(opponent_generated_logps:\s*tensor\([^\)]*?\))"

def draw_line_plot(file_path, marker, tag):
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
            'device': int(blocks[0][0].split('device=\'cuda:')[-1].split("'")[0])
        })
        


    df = pd.DataFrame(data)


    # Calculate the 5th and 95th percentiles for the 'X' column
    x_lower, x_upper = np.percentile(df['policy_real_logp'], [5, 95])
    # Filter the DataFrame to keep only the rows where 'X' is within this range
    df_filtered = df[(df['policy_real_logp'] >= x_lower) & (df['policy_real_logp'] <= x_upper)]
    y_lower, y_upper = np.percentile(df['policy_generated_logp'], [5, 95])
    df_filtered = df_filtered[(df_filtered['policy_generated_logp'] >= y_lower) & (df_filtered['policy_generated_logp'] <= y_upper)]

    # Create a plot
    step = 2000
    plot_data = []
    for start in range(0, df.shape[0],step):
        policy_real_logp = df_filtered['policy_real_logp'][start:start+step].mean()
        policy_generated_logp = df_filtered['policy_generated_logp'][start:start+step].mean()
        opponent_real_logp = df_filtered['opponent_real_logp'][start:start+step].mean()
        opponent_generated_logp = df_filtered['opponent_generated_logp'][start:start+step].mean()
        
        plot_data.append({'policy_real_logp': policy_real_logp, 'policy_generated_logp': policy_generated_logp, 'opponent_real_logp': opponent_real_logp, 'opponent_generated_logp': opponent_generated_logp})
        
    df_plot = pd.DataFrame(plot_data)


    plt.plot(df_plot['policy_real_logp'] , marker=marker, linestyle='-', color='b',label=f'{tag}_policy_winner_logp') # 'o' for markers, '-' for lines
    plt.plot(df_plot['policy_generated_logp'] , marker=marker, linestyle='-', color='r',label=f'{tag}_policy_loser_logp') 
    plt.plot(df_plot['opponent_real_logp'] , marker=marker, linestyle='-', color='g',label=f'{tag}_reference_winner_logp') # 
    plt.plot(df_plot['opponent_generated_logp'] , marker=marker, linestyle='-', color='y',label=f'{tag}_reference_loser_logp') # 






file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/NoneConfig_stage1_800_spin_256_SPIN_iter0_L40_1e-5__725725.out'
file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_800_SPIN_iter1_L40_1e-5__728816.out'
only_lora_file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_1024_lora_only_SPIN_iter0_L40_1e-5__732983.out'
only_lora_file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/stage1_1024_lora_only_SPIN_iter1_L40_1e-5__732984.out'
beta_lora_file_path_1 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/beta_1e-2_stage1_1024_lora_only_SPIN_iter1_L40_1e-5__732985.out'
beta_lora_file_path_0 = '/home/srijithr/iterative-alignment/gradient_field/out_log_from_runs/beta_1e-2_stage1_1024_lora_only_SPIN_iter0_L40_1e-5__732986.out'


plt.figure(figsize=(8, 5))  
# draw_line_plot(file_path_0, 'o', 'iter0')
draw_line_plot(only_lora_file_path_1, 'x', 'iter1')


plt.tight_layout()
plt.grid()
plt.legend(loc='lower left')
plt.savefig('/home/srijithr/iterative-alignment/gradient_field/images/only_lora_file_path_1.png', dpi= 500)