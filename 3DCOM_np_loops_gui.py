import matplotlib
matplotlib.use('TkAgg')
import tkinter as tk
from tkinter import filedialog,messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


# --- Collatz & Octave Model logic ---
def generate_collatz_sequence(n, max_steps=800):
    sequence = [n]
    seen = {n: 0}
    curr = n
    for step in range(1, max_steps):
        if curr == 1:  # positive side terminal
            break
        if curr % 2 == 0:
            curr = curr // 2
        else:
            curr = 3 * curr + 1
        if curr in seen:
            # Cycle detected
            cycle_start = seen[curr]
            cycle = sequence[cycle_start:]
            return sequence, tuple(cycle)
        sequence.append(curr)
        seen[curr] = len(sequence) - 1
    return sequence, None  # No cycle (escaped, or positive n)


def reduce_to_single_digit(value):
    return (abs(value) - 1) % 9 + 1


def map_to_octave(val, layer):
    angle = (val / 9) * 2 * np.pi
    x = np.cos(angle) * (layer + 1)
    y = np.sin(angle) * (layer + 1)
    return x, y

# --- GUI Application ---


class Collatz3DCOMApp(tk.Tk):
    NEGATIVE_CYCLE_COLORS = [
        "orange", "violet", "darkcyan", "lime", "gold", "magenta", "brown", "teal", "orchid", "navy"
    ]

    def __init__(self):
        super().__init__()
        self.title("Collatz 3DCOM: Octave Geometry + Negative Cycle Detection")
        self.geometry("1200x900")

        frm = tk.Frame(self)
        frm.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        tk.Label(frm, text="Negative n: from -").pack(side=tk.LEFT)
        self.neg_from = tk.Entry(frm, width=5)
        self.neg_from.insert(0, "20")
        self.neg_from.pack(side=tk.LEFT)
        tk.Label(frm, text="to -").pack(side=tk.LEFT)
        self.neg_to = tk.Entry(frm, width=5)
        self.neg_to.insert(0, "1")
        self.neg_to.pack(side=tk.LEFT)
        tk.Label(frm, text="   Positive n: from ").pack(side=tk.LEFT)
        self.pos_from = tk.Entry(frm, width=5)
        self.pos_from.insert(0, "1")
        self.pos_from.pack(side=tk.LEFT)
        tk.Label(frm, text="to ").pack(side=tk.LEFT)
        self.pos_to = tk.Entry(frm, width=5)
        self.pos_to.insert(0, "20")
        self.pos_to.pack(side=tk.LEFT)
        tk.Button(frm, text="Show Negatives (cycle colors)",
                  command=self.plot_negatives_cycles).pack(side=tk.LEFT, padx=3)
        tk.Button(frm, text="Show Positives", command=self.plot_positives).pack(
            side=tk.LEFT, padx=3)
        tk.Button(frm, text="Overlap Both", command=self.plot_overlap).pack(
            side=tk.LEFT, padx=3)
        tk.Button(frm, text="Export Data", command=self.export_data).pack(
            side=tk.LEFT, padx=3)

        self.fig = plt.figure(figsize=(12, 9))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self)
        self.toolbar.update()
        self.toolbar.pack()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.last_plot_data = {}

    def get_ranges(self):
        try:
            nf = abs(int(self.neg_from.get()))
            nt = abs(int(self.neg_to.get()))
        except Exception:
            nf, nt = 20, 1
        try:
            pf = int(self.pos_from.get())
            pt = int(self.pos_to.get())
        except Exception:
            pf, pt = 1, 20
        neg_range = range(-nf, -nt-1, 1) if nf >= nt else []
        pos_range = range(pf, pt+1, 1) if pf <= pt else []
        return neg_range, pos_range

    def gather_octave_data_negative_cycles(self, n_range):
        cycle_dict = {}      # key: tuple(cycle), value: index/color
        cycle_members = {}   # key: tuple(cycle), value: set(all seen members)
        seq_data = {}        # key: n, value: (seq, cycle tuple or None)
        for n in n_range:
            seq, cycle = generate_collatz_sequence(n)
            seq_data[n] = (seq, cycle)
            if cycle:
                c_key = tuple(sorted(set(cycle)))
                if c_key not in cycle_dict:
                    cycle_dict[c_key] = len(cycle_dict) % len(
                        self.NEGATIVE_CYCLE_COLORS)
                    cycle_members[c_key] = set(cycle)
        return seq_data, cycle_dict

    def plot_negatives_cycles(self):
        neg_range, _ = self.get_ranges()
        self.ax.clear()
        seq_data, cycle_dict = self.gather_octave_data_negative_cycles(
            neg_range)
        for n, (seq, cycle) in seq_data.items():
            xs, ys, zs = [], [], []
            colors = []
            in_cycle = False
            c_idx = None
            if cycle:
                c_key = tuple(sorted(set(cycle)))
                if c_key in cycle_dict:
                    c_idx = cycle_dict[c_key]
            for i, val in enumerate(seq):
                x, y = map_to_octave(reduce_to_single_digit(val), i)
                xs.append(x)
                ys.append(y)
                zs.append(i)
                if cycle and val in cycle:
                    in_cycle = True
                else:
                    in_cycle = False
                if in_cycle and c_idx is not None:
                    colors.append(self.NEGATIVE_CYCLE_COLORS[c_idx])
                else:
                    colors.append("red")
            # Plot the full sequence line (mainly for context)
            self.ax.plot(xs, ys, zs, color="grey", alpha=0.3, linewidth=1)
            # Overlay just the cycle loop in bright color if present
            if cycle and c_idx is not None:
                # Find where the cycle starts/ends and plot in thick color
                cycle_idxs = [i for i, val in enumerate(seq) if val in cycle]
                for k in range(len(cycle_idxs)-1):
                    i1, i2 = cycle_idxs[k], cycle_idxs[k+1]
                    self.ax.plot(xs[i1:i2+1], ys[i1:i2+1], zs[i1:i2+1],
                                 color=self.NEGATIVE_CYCLE_COLORS[c_idx], alpha=0.96, linewidth=3)
        self.ax.set_title(
            "Negative n: Collatz Cycles Highlighted (Standard Rule)")
        self.ax.set_xlabel("X octave")
        self.ax.set_ylabel("Y octave")
        self.ax.set_zlabel("Z: layer/step")
        # Build/Show legend
        handles = []
        labels = []
        for cyc, idx in cycle_dict.items():
            handles.append(plt.Line2D(
                [0], [0], color=self.NEGATIVE_CYCLE_COLORS[idx], lw=4))
            labels.append(f"Cycle {list(cyc)}")
        if handles:
            self.ax.legend(handles, labels, loc='upper left', fontsize='small')
        self.canvas.draw()
        self.last_plot_data = {"neg_cycles": seq_data}

    def plot_positives(self):
        _, pos_range = self.get_ranges()
        self.ax.clear()
        for n in pos_range:
            seq, _ = generate_collatz_sequence(n)
            xs, ys, zs = [], [], []
            for i, v in enumerate(seq):
                x, y = map_to_octave(reduce_to_single_digit(v), i)
                xs.append(x)
                ys.append(y)
                zs.append(i)
            self.ax.plot(xs, ys, zs, color='deepskyblue', alpha=0.85)
        self.ax.set_title(
            "Positive n: Collatz 3DCOM Octave Geometry (Converging to 1)")
        self.ax.set_xlabel("X octave")
        self.ax.set_ylabel("Y octave")
        self.ax.set_zlabel("Z: layer/step")
        self.canvas.draw()
        self.last_plot_data = {"pos": pos_range}

    def plot_overlap(self):
        neg_range, pos_range = self.get_ranges()
        self.ax.clear()
        # Negatives with cycle coloring
        seq_data, cycle_dict = self.gather_octave_data_negative_cycles(
            neg_range)
        for n, (seq, cycle) in seq_data.items():
            xs, ys, zs = [], [], []
            c_idx = None
            if cycle:
                c_key = tuple(sorted(set(cycle)))
                if c_key in cycle_dict:
                    c_idx = cycle_dict[c_key]
            # Plot full sequence
            for i, v in enumerate(seq):
                x, y = map_to_octave(reduce_to_single_digit(v), i)
                xs.append(x)
                ys.append(y)
                zs.append(i)
            # Draw main path in light grey
            self.ax.plot(xs, ys, zs, color="grey", alpha=0.3, linewidth=1)
            # Draw loop part bold if present
            if cycle and c_idx is not None:
                cycle_idxs = [i for i, val in enumerate(seq) if val in cycle]
                for k in range(len(cycle_idxs)-1):
                    i1, i2 = cycle_idxs[k], cycle_idxs[k+1]
                    self.ax.plot(xs[i1:i2+1], ys[i1:i2+1], zs[i1:i2+1],
                                 color=self.NEGATIVE_CYCLE_COLORS[c_idx], alpha=0.96, linewidth=3)
        # Positives
        for n in pos_range:
            seq, _ = generate_collatz_sequence(n)
            xs, ys, zs = [], [], []
            for i, v in enumerate(seq):
                x, y = map_to_octave(reduce_to_single_digit(v), i)
                xs.append(x)
                ys.append(y)
                zs.append(i)
            self.ax.plot(xs, ys, zs, color='deepskyblue',
                         alpha=0.75, linewidth=2)
        self.ax.set_title(
            "Collatz 3DCOM: Positive (blue) and Negative Cycles (unique colors)")
        self.ax.set_xlabel("X octave")
        self.ax.set_ylabel("Y octave")
        self.ax.set_zlabel("Z: layer/step")
        handles = []
        labels = []
        for cyc, idx in cycle_dict.items():
            handles.append(plt.Line2D(
                [0], [0], color=self.NEGATIVE_CYCLE_COLORS[idx], lw=4))
            labels.append(f"Cycle {list(cyc)}")
        if handles:
            handles.append(plt.Line2D([0], [0], color="deepskyblue", lw=4))
            labels.append("Positive n")
            self.ax.legend(handles, labels, loc='upper left', fontsize='small')
        self.canvas.draw()
        self.last_plot_data = {"neg_cycles": seq_data, "pos": pos_range}

    def export_data(self):
        fpath = filedialog.asksaveasfilename(defaultextension=".csv")
        if not fpath or not self.last_plot_data:
            return
        with open(fpath, "w") as f:
            # Export detected negative cycles if present
            if "neg_cycles" in self.last_plot_data:
                seq_data = self.last_plot_data["neg_cycles"]
                for n, (seq, cycle) in seq_data.items():
                    for i, val in enumerate(seq):
                        r = reduce_to_single_digit(val)
                        x, y = map_to_octave(r, i)
                        z = i
                        cyc_lab = f"CYCLE_{cycle}" if cycle and val in cycle else ""
                        f.write(
                            f"neg,{n},{i},{val},{x:.4f},{y:.4f},{z},{cyc_lab}\n")
            # Export positives
            if "pos" in self.last_plot_data:
                pos_range = self.last_plot_data["pos"]
                for n in pos_range:
                    seq, _ = generate_collatz_sequence(n)
                    for i, val in enumerate(seq):
                        r = reduce_to_single_digit(val)
                        x, y = map_to_octave(r, i)
                        z = i
                        f.write(f"pos,{n},{i},{val},{x:.4f},{y:.4f},{z},\n")
        self.fig.savefig(fpath.replace('.csv', '.png'), dpi=200)


# --- RUN ---
if __name__ == "__main__":
    app = Collatz3DCOMApp()
    app.mainloop()
