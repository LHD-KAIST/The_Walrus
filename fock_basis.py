# This code show us to plot any non-fock state in fock-basis.
import matplotlib.pyplot as plt
from pathlib import Path

import strawberryfields as sf
from strawberryfields.plot import generate_fock_chart
from strawberryfields.ops import *

prog = sf.Program(1)

with prog.context as q:
    Fock(0) | q[0]
    sf.ops.Dgate(4) | q[0]


eng = sf.Engine('fock', backend_options={"cutoff_dim": 20})
result = eng.run(prog)

modes_to_plot = [0]
cutoff = 20

plots_dir = Path(__file__).resolve().parent / "plots"
plots_dir.mkdir(parents=True, exist_ok=True)

chart = generate_fock_chart(result.state, modes_to_plot, cutoff)

# 결과 이미지를 저장할 디렉터리를 준비
num_modes = len(modes_to_plot)
fig, axes = plt.subplots(num_modes, 1, figsize=(8, 3 * num_modes), squeeze=False)

# Plotly 데이터에서 확률 분포를 꺼내 모드별 막대그래프를 그림
for ax, mode, data in zip(axes.flat, sorted(modes_to_plot), chart["data"]):
    probabilities = data["y"]
    labels = [f"|{n}>" for n in range(len(probabilities))]

    ax.bar(range(len(probabilities)), probabilities, color="#1f9094")
    ax.set_title(f"mode {mode}")
    ax.set_xlabel("Fock state")
    ax.set_ylabel("Probability")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    upper = max(probabilities) * 1.1 if probabilities and max(probabilities) > 0 else 1
    ax.set_ylim(0, upper)

# 레이아웃을 다듬고 PNG 파일로 저장
fig.tight_layout()

output_path = plots_dir / "fock_chart.png"
fig.savefig(output_path, bbox_inches="tight")
plt.close(fig)
