import json
from pathlib import Path
import matplotlib.pyplot as plt

# Plot only: read JSON -> output PNG. No changes to env/controller logic.

ROOT = Path(__file__).resolve().parents[1]
json_path = ROOT / "analysis" / "WEEK7_ablation_results.json"
out_path = ROOT / "analysis" / "fig_sat_rate_raw_vs_prescale.png"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

modes = ["raw", "prescale"]
controllers = ["ExpertV3", "ExpertImproved"]

# y[controller][mode]
y = {c: [data["results"][m][c]["saturation_rate"] for m in modes] for c in controllers}

x = [0, 1]
x_v3 = [xi - 0.02 for xi in x]
x_imp = [xi + 0.02 for xi in x]
plt.plot(x_v3, y["ExpertV3"], marker="o", label="ExpertV3")
plt.plot(x_imp, y["ExpertImproved"], marker="o", label="ExpertImproved")


plt.figure()
for c in controllers:
    plt.plot(x, y[c], marker="o", label=c)

plt.xticks(list(x), modes)
plt.xlabel("ACTION_IF_MODE")
plt.ylabel("saturation_rate")
plt.title("Week7 Ablation: saturation_rate (raw vs prescale)")
plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"[OK] Saved: {out_path}")
