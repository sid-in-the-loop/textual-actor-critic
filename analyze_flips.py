import json
import matplotlib.pyplot as plt
import numpy as np
import os

BATCH_SIZE = 128
FILE_PATH = 'logs/trajectories/current_rollout_2.jsonl'

def parse_data():
    steps_1_15 = []
    steps_1_1 = []
    
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} not found.")
        return [], []
        
    with open(FILE_PATH, 'r') as f:
        lines = f.readlines()
        
    num_batches = len(lines) // BATCH_SIZE
    print(f"Processing {num_batches} batches of {BATCH_SIZE} samples each...")
    
    for i in range(num_batches):
        batch = lines[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        flips = {'c_to_i': 0, 'i_to_c': 0, 'i_to_i': 0}
        
        for line in batch:
            try:
                data = json.loads(line)
                t1 = data.get('turn1_correct', False)
                t3 = data.get('turn3_correct', False)
                
                if t1 and not t3:
                    flips['c_to_i'] += 1
                elif not t1 and t3:
                    flips['i_to_c'] += 1
                elif not t1 and not t3:
                    flips['i_to_i'] += 1
            except json.JSONDecodeError:
                continue
        
        # Convert to percentages
        stats = {k: (v / BATCH_SIZE) * 100 for k, v in flips.items()}
        
        # Alternating: 1_15 then 1_1
        if i % 2 == 0:
            steps_1_15.append(stats)
        else:
            steps_1_1.append(stats)
            
    return steps_1_15, steps_1_1

def plot(steps_1_15, steps_1_1):
    # Set up high-res plot
    plt.rcParams.update({'font.size': 12})
    fig, ax1 = plt.subplots(figsize=(19.2, 10.8), dpi=1600)
    ax2 = ax1.twinx()
    
    # Common X-axis (Training Steps)
    num_steps = min(len(steps_1_15), len(steps_1_1))
    x = np.arange(num_steps)
    
    # Extract metrics for plotting
    c_to_i_15 = [s['c_to_i'] for s in steps_1_15[:num_steps]]
    i_to_c_15 = [s['i_to_c'] for s in steps_1_15[:num_steps]]
    i_to_i_15 = [s['i_to_i'] for s in steps_1_15[:num_steps]]
    
    c_to_i_1 = [s['c_to_i'] for s in steps_1_1[:num_steps]]
    i_to_c_1 = [s['i_to_c'] for s in steps_1_1[:num_steps]]
    i_to_i_1 = [s['i_to_i'] for s in steps_1_1[:num_steps]]
    
    # --- MODEL 1_15 (Bold/Solid Lines) ---
    # Left Axis: Delta C to I (Blue) and Remaining I (Green)
    ax1.plot(x, c_to_i_15, color='blue', linewidth=3.5, label=r'1_15: $\Delta^{c \to i}$')
    ax1.plot(x, i_to_i_15, color='green', linewidth=3.5, label=r'1_15: $i \to i$')
    # Right Axis: Delta I to C (Orange)
    ax2.plot(x, i_to_c_15, color='orange', linewidth=3.5, label=r'1_15: $\Delta^{i \to c}$')
    
    # --- MODEL 1_1 (Dotted Lines) ---
    ax1.plot(x, c_to_i_1, color='blue', linestyle=':', linewidth=2.0, label=r'1_1: $\Delta^{c \to i}$')
    ax1.plot(x, i_to_i_1, color='green', linestyle=':', linewidth=2.0, label=r'1_1: $i \to i$')
    ax2.plot(x, i_to_c_1, color='orange', linestyle=':', linewidth=2.0, label=r'1_1: $\Delta^{i \to c}$')
    
    # Axis Customization
    ax1.set_xlabel('Training Steps', fontsize=16, fontweight='bold')
    ax1.set_ylabel(r'$\Delta^{c \to i}$ / $i \to i$ (%)', fontsize=16, fontweight='bold', color='black')
    ax2.set_ylabel(r'$\Delta^{i \to c}$ (%)', fontsize=16, fontweight='bold', color='orange')
    
    # Scales 0-100
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    
    # Grid and Title
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.title('Correctness Flips Analysis: Turn 1 vs Turn 3', fontsize=20, fontweight='bold', pad=20)
    
    # Unified Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.1), 
               ncol=3, fontsize=12, frameon=True, shadow=True)
    
    plt.tight_layout()
    output_fn = 'correctness_flips_hd.png'
    plt.savefig(output_fn, dpi=1600, bbox_inches='tight')
    print(f"✅ Full HD Plot saved to {output_fn} (DPI: 1600)")

if __name__ == "__main__":
    print(f"Analyzing trajectory file: {FILE_PATH}")
    s15, s1 = parse_data()
    if s15 and s1:
        plot(s15, s1)
    else:
        print("No data found to plot.")

