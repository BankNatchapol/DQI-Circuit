import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create figures folder if it doesn't exist
os.makedirs("figures", exist_ok=True)

# Set global font size and line width for better readability
sns.set(style="whitegrid", context="talk", font_scale=1.4)
plt.rcParams.update({
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "axes.labelsize": 25,
    "axes.titlesize": 25,
    "legend.fontsize": 16,
    "legend.title_fontsize": 18,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20
})

decode_methods = ["GJE", "Lookup Table"]

metrics = {
    "One Qubit Gates": "one_qubit_gates",
    "Two Qubit Gates": "two_qubit_gates",
    "Total Gates": "total_gates",
    "Depth": "depth"
}

# -------------------------------
# Helper Function to Plot (without Lookup Table)
# -------------------------------
def plot_metrics(entries, x_key, x_label, filename_prefix, fixed_note=""):
    # Add Hadamard Transform to the parts list so its data gets plotted.
    parts_list = [
        "Initialization Circuit",
        "Dicke State Circuit",
        "Phase Flip Circuit",
        "Constraint Encoding Circuit",
        "Decoding.GJE",
        "Hadamard Transform",
        "Complete DQI Circuit (GJE)",
        
    ]

    for metric_name, metric_key in metrics.items():
        rows = []
        for entry in entries:
            x_val = entry.get(x_key)
            for part in parts_list:
                if part.startswith("Decoding."):
                    part_data = entry.get("Decoding", {}).get(part.split(".")[1], {})
                    label = "Decoding (GJE)"
                else:
                    part_data = entry.get(part, {})
                    label = part
                rows.append({
                    x_label: x_val,
                    "Circuit Part": label,
                    metric_name: part_data.get(metric_key, 0)
                })

        df = pd.DataFrame(rows)
        plt.figure(figsize=(12, 8))
        ax = sns.lineplot(data=df, x=x_label, y=metric_name, hue="Circuit Part", marker="o")
        ax.set_title(f"{metric_name} by {x_label} {fixed_note}")
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric_name)
        plt.xticks(rotation=45 if x_key == "Instance" else 0)
        plt.legend(title="Circuit Part")
        plt.tight_layout()
        plt.savefig(f"figures/{filename_prefix}_{metric_key}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: figures/{filename_prefix}_{metric_key}.png")

# -------------------------------
# Plot Decoding Comparison (GJE vs Lookup Table)
# -------------------------------
def plot_decoding_comparison(entries, x_key, x_label, filename_prefix, fixed_note=""):
    for metric_name, metric_key in metrics.items():
        rows = []
        for entry in entries:
            x_val = entry.get(x_key)
            for method in decode_methods:
                decoding_data = entry.get("Decoding", {}).get(method, {})
                rows.append({
                    x_label: x_val,
                    "Decoding Method": method,
                    metric_name: decoding_data.get(metric_key, 0)
                })

        df = pd.DataFrame(rows)
        plt.figure(figsize=(12, 8))
        ax = sns.lineplot(data=df, x=x_label, y=metric_name, hue="Decoding Method", marker="o")
        ax.set_title(f"Decoding: {metric_name} by {x_label} {fixed_note}", fontsize=18)
        ax.set_xlabel(x_label, fontsize=16)
        ax.set_ylabel(metric_name, fontsize=16)
        plt.xticks(rotation=45 if x_key == "Instance" else 0)
        plt.legend(title="Decoding Method", fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.savefig(f"figures/{filename_prefix}_decoding_{metric_key}.png", bbox_inches="tight")
        plt.close()
        print(f"Saved: figures/{filename_prefix}_decoding_{metric_key}.png")

# -------------------------------
# Load and Plot info_fix_ell.json
# -------------------------------
with open("info_fix_ell.json", "r") as f:
    data_inst = json.load(f)

benchmark_inst = data_inst.get("benchmark", [])
if isinstance(benchmark_inst, dict):
    benchmark_inst = list(benchmark_inst.values())

for entry in benchmark_inst:
    entry["Instance"] = entry.get("instance", f"{entry.get('num_nodes')}x{entry.get('num_edges')}")

first_inst = benchmark_inst[0]
plot_metrics(
    entries=sorted(benchmark_inst, key=lambda x: int(x["Instance"].split("x")[0])),
    x_key="Instance",
    x_label="Instance",
    filename_prefix="inst",
    fixed_note=f"(Fixed: p={first_inst['p']}, r={first_inst['r']}, ℓ={first_inst['ell']})"
)

plot_decoding_comparison(
    entries=sorted(benchmark_inst, key=lambda x: int(x["Instance"].split("x")[0])),
    x_key="Instance",
    x_label="Instance",
    filename_prefix="inst",
    fixed_note=f"(Fixed: p={first_inst['p']}, r={first_inst['r']}, ℓ={first_inst['ell']})"
)

# -------------------------------
# Load and Plot info_fix_B.json
# -------------------------------
with open("info_fix_B.json", "r") as f:
    data_ell = json.load(f)

benchmark_ell = data_ell.get("benchmark", [])
if isinstance(benchmark_ell, dict):
    benchmark_ell = list(benchmark_ell.values())

first_ell = benchmark_ell[0]
plot_metrics(
    entries=sorted(benchmark_ell, key=lambda x: x["ell"]),
    x_key="ell",
    x_label="ℓ",
    filename_prefix="ell",
    fixed_note=f"(Fixed: Instance=12x12, p={first_ell['p']}, r={first_ell['r']})"
)

plot_decoding_comparison(
    entries=sorted(benchmark_ell, key=lambda x: x["ell"]),
    x_key="ell",
    x_label="ℓ",
    filename_prefix="ell",
    fixed_note=f"(Fixed: Instance=12x12, p={first_ell['p']}, r={first_ell['r']})"
)
