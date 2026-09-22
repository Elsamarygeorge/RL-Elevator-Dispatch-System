"""
visualizer.py
Tkinter-based GUI for the Multi-Elevator Dispatch project.

Location: frontend/visualizer.py  (sibling to backend/)
Run from the project ROOT folder:
    python frontend/visualizer.py

No extra installs needed — tkinter ships with Python's standard library.

Requires backend/rl/env.py's reset() to accept an optional seed parameter
(already true in Shreya's final version) and backend/rl/q_table.pkl to
exist (the trained Q-table, saved by test_q_learning.py).
"""

import sys
import os
import random
import tkinter as tk
from tkinter import ttk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from rl.env import ElevatorEnv
from rl.q_learning import QLearningAgent
from algorithms.nearest import NearestElevatorDispatcher
from algorithms.first_available import FirstAvailableDispatcher
from algorithms.round_robin import RoundRobinDispatcher
from algorithms.random_assign import RandomDispatcher
from config import NUM_FLOORS, NUM_ELEVATORS, SIMULATION_STEPS

# ---------------------------------------------------------------
# Shared styling
# ---------------------------------------------------------------
BG_DARK = "#181a20"
BG_PANEL = "#22242c"
BG_SHAFT = "#262933"
FG_TEXT = "#e6e6eb"
FG_MUTED = "#9a9ca6"
COLOR_IDLE = "#5a966a"
COLOR_MOVING = "#5a82dc"
COLOR_FULL = "#dc645a"
COLOR_WAITING = "#f0c850"
FONT_NORMAL = ("Consolas", 11)
FONT_BOLD = ("Consolas", 11, "bold")
FONT_HEADER = ("Consolas", 14, "bold")

SIDEBAR_TITLE = "RL-BASED ELEVATOR DISPATCH SYSTEM"
SECONDS_PER_STEP = 3
SPEED_LEVELS = [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50]

# Q-Learning listed first and used as the default everywhere — it's the
# project's actual headline result, not just another baseline to compare.
STRATEGY_NAMES = [
    "Q-Learning (Trained)",
    "Nearest Elevator",
    "First Available",
    "Round Robin",
    "Random",
]

# ---------------------------------------------------------------
# Trained agent — loaded once, reused everywhere.
# ---------------------------------------------------------------
_trained_agent = None


def get_trained_agent():
    global _trained_agent
    if _trained_agent is None:
        _trained_agent = QLearningAgent()
        _trained_agent.load_q_table()
        _trained_agent.epsilon = 0.0  # greedy — no exploration during demo/comparison
    return _trained_agent


class QLearningDispatcher:
    """
    Adapts the trained QLearningAgent (which works on an ENCODED STATE via
    choose_action(state)) to the same interface every other dispatcher in
    this file uses: choose_action(building). It asks the ElevatorEnv it's
    bound to for its own current encoded state, then hands that to the
    trained agent.
    """
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.__class__.__name__ = "QLearningDispatcher"

    def choose_action(self, building):
        state = self.env._encode_state()
        return self.agent.choose_action(state)


def create_dispatcher(name, env):
    """Builds the dispatcher for a given strategy name. Needs `env` because
    the Q-Learning strategy has to read that specific environment's state."""
    if name == "Q-Learning (Trained)":
        return QLearningDispatcher(env, get_trained_agent())
    if name == "Nearest Elevator":
        return NearestElevatorDispatcher()
    if name == "First Available":
        return FirstAvailableDispatcher()
    if name == "Round Robin":
        return RoundRobinDispatcher(NUM_ELEVATORS)
    if name == "Random":
        return RandomDispatcher(NUM_ELEVATORS)
    raise ValueError(f"Unknown strategy: {name}")


def random_seed_value():
    return random.randint(0, 2**31 - 1)


def choose_action_safely(dispatcher, building):
    """
    Only asks the dispatcher for an action when there's an actual pending
    request to decide on. Required for correctness with stateful
    dispatchers like RoundRobinDispatcher — calling choose_action() on
    every tick (even no-op ticks) would silently advance its internal
    counter on wasted calls, desyncing its assignment rotation.
    """
    if building.next_waiting_request() is None:
        return 0
    return dispatcher.choose_action(building)


def waiting_by_floor(building):
    """Passengers physically standing at each floor right now — both
    unassigned requests and requests already assigned but not yet boarded."""
    counts = {}
    for req in building.waiting_requests:
        f = req.passenger.source_floor
        counts[f] = counts.get(f, 0) + 1
    for e in building.elevators:
        for req in e.assigned_requests:
            f = req.passenger.source_floor
            counts[f] = counts.get(f, 0) + 1
    return counts


def total_waiting(building):
    return sum(waiting_by_floor(building).values())


def run_full_episode(strategy_name, seed):
    """Run one complete simulated day with a strategy, no animation."""
    env = ElevatorEnv()
    env.reset(seed=seed)
    dispatcher = create_dispatcher(strategy_name, env)
    total_reward = 0.0
    done = False
    while not done:
        action = choose_action_safely(dispatcher, env.building)
        _, reward, done, _ = env.step(action)
        total_reward += reward

    completed = env.building.completed_count
    total_wait_time = sum(r.passenger.waiting_time for r in env.building.completed_requests)
    avg_wait = total_wait_time / completed if completed else 0
    return {
        "completed": completed,
        "waiting": total_waiting(env.building),
        "avg_wait": avg_wait,
        "reward": total_reward,
    }


# =================================================================
# Live View page
# =================================================================
class LivePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_DARK)
        self.app = app

        self.env = ElevatorEnv()
        self.env.reset(seed=self.app.current_seed)
        self.dispatcher = create_dispatcher("Q-Learning (Trained)", self.env)
        self.step_count = 0
        self.total_reward = 0.0
        self.running = False
        self.speed_index = SPEED_LEVELS.index(2)

        self.displayed_floor = [float(e.current_floor) for e in self.env.building.elevators]
        self.target_floor = [float(e.current_floor) for e in self.env.building.elevators]

        self._build_layout()
        self._animate()

    # ---------------- layout ----------------
    def _build_layout(self):
        info_frame = tk.Frame(self, bg=BG_PANEL)
        info_frame.pack(side="top", fill="x")
        self.info_var = tk.StringVar()
        tk.Label(
            info_frame, textvariable=self.info_var, bg=BG_PANEL, fg=FG_TEXT,
            font=FONT_BOLD, anchor="w", padx=10, pady=8,
        ).pack(side="left", fill="x", expand=True)

        controls = tk.Frame(self, bg=BG_DARK)
        controls.pack(side="top", fill="x", pady=4)
        self.strategy_var = tk.StringVar(value="Q-Learning (Trained)")
        strategy_menu = ttk.Combobox(
            controls, textvariable=self.strategy_var,
            values=STRATEGY_NAMES, state="readonly", width=20,
        )
        strategy_menu.pack(side="left", padx=6)
        strategy_menu.bind("<<ComboboxSelected>>", lambda e: self._change_strategy())

        tk.Button(controls, text="Start/Pause", command=self._toggle_running).pack(side="left", padx=4)
        tk.Button(controls, text="Reset (same day)", command=self._reset).pack(side="left", padx=4)
        tk.Button(controls, text="New Day", command=self._new_day).pack(side="left", padx=4)
        tk.Button(controls, text="Speed -", command=lambda: self._change_speed(-1)).pack(side="left", padx=4)
        tk.Button(controls, text="Speed +", command=lambda: self._change_speed(1)).pack(side="left", padx=4)
        self.speed_label = tk.Label(controls, text="", bg=BG_DARK, fg=FG_MUTED, font=FONT_NORMAL)
        self.speed_label.pack(side="left", padx=8)

        body = tk.Frame(self, bg=BG_DARK)
        body.pack(side="top", fill="both", expand=True)

        self.canvas = tk.Canvas(body, bg=BG_DARK, highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)

        self.floor_panel = tk.Frame(body, bg=BG_PANEL, width=180)
        self.floor_panel.pack(side="right", fill="y")
        self.floor_labels = {}
        tk.Label(self.floor_panel, text="Floor Status", bg=BG_PANEL, fg=FG_TEXT, font=FONT_BOLD).pack(pady=(10, 6))
        for floor in range(NUM_FLOORS, 0, -1):
            lbl = tk.Label(
                self.floor_panel, text=f"F{floor}: —", bg=BG_PANEL, fg=FG_MUTED,
                font=FONT_NORMAL, anchor="w", width=20,
            )
            lbl.pack(fill="x", padx=10, pady=1)
            self.floor_labels[floor] = lbl

        self._update_speed_label()

    # ---------------- controls ----------------
    def _toggle_running(self):
        self.running = not self.running

    def _reset(self):
        self.env.reset(seed=self.app.current_seed)
        self.step_count = 0
        self.total_reward = 0.0
        self.running = False
        self.displayed_floor = [float(e.current_floor) for e in self.env.building.elevators]
        self.target_floor = [float(e.current_floor) for e in self.env.building.elevators]

    def _new_day(self):
        self.app.current_seed = random_seed_value()
        self._reset()

    def _change_strategy(self):
        self.dispatcher = create_dispatcher(self.strategy_var.get(), self.env)
        self._reset()

    def _change_speed(self, direction):
        self.speed_index = max(0, min(len(SPEED_LEVELS) - 1, self.speed_index + direction))
        self._update_speed_label()

    @property
    def sim_speed(self):
        return SPEED_LEVELS[self.speed_index]

    def _update_speed_label(self):
        self.speed_label.config(text=f"Speed: {self.sim_speed}x")

    # ---------------- simulation + animation loop ----------------
    def _simulation_tick(self):
        if self.step_count >= SIMULATION_STEPS:
            self.running = False
            return
        action = choose_action_safely(self.dispatcher, self.env.building)
        _, reward, done, _ = self.env.step(action)
        self.total_reward += reward
        self.step_count += 1
        self.target_floor = [float(e.current_floor) for e in self.env.building.elevators]
        if done:
            self.running = False

    def _animate(self):
        self._frame_accum = getattr(self, "_frame_accum", 0.0) + 1.0 / 30.0
        seconds_per_step = 1.0 / self.sim_speed
        if self.running and self._frame_accum >= seconds_per_step:
            self._frame_accum = 0.0
            self._simulation_tick()

        for i in range(len(self.displayed_floor)):
            diff = self.target_floor[i] - self.displayed_floor[i]
            if abs(diff) > 0.01:
                self.displayed_floor[i] += diff * 0.25

        self._draw()
        self.after(33, self._animate)

    # ---------------- drawing ----------------
    def _draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width = canvas.winfo_width() or 800
        height = canvas.winfo_height() or 600
        margin_top, margin_bottom = 20, 20
        floor_h = (height - margin_top - margin_bottom) / NUM_FLOORS
        shaft_w = 110
        gap = 30
        left_pad = 60

        def floor_y(f):
            return margin_top + (NUM_FLOORS - f) * floor_h

        for f in range(1, NUM_FLOORS + 1):
            y = floor_y(f)
            canvas.create_line(0, y + floor_h, width, y + floor_h, fill="#33363f")
            canvas.create_text(15, y + floor_h / 2, text=f"F{f}", fill=FG_MUTED, font=FONT_NORMAL, anchor="w")

        counts = waiting_by_floor(self.env.building)
        for floor in range(1, NUM_FLOORS + 1):
            count = counts.get(floor, 0)
            self.floor_labels[floor].config(
                text=f"F{floor}: {count} waiting" if count else f"F{floor}: —",
                fg=COLOR_WAITING if count else FG_MUTED,
            )

        for i, e in enumerate(self.env.building.elevators):
            shaft_x = left_pad + i * (shaft_w + gap)
            canvas.create_rectangle(
                shaft_x, margin_top, shaft_x + shaft_w, height - margin_bottom,
                fill=BG_SHAFT, outline=""
            )
            canvas.create_text(
                shaft_x + shaft_w / 2, margin_top - 10, text=f"Elevator {i}",
                fill=FG_TEXT, font=FONT_BOLD
            )

            car_h = 90
            car_y = floor_y(self.displayed_floor[i]) + (floor_h - car_h) / 2
            if e.is_full():
                color = COLOR_FULL
            elif e.status.value == "MOVING":
                color = COLOR_MOVING
            else:
                color = COLOR_IDLE
            canvas.create_rectangle(
                shaft_x + 5, car_y, shaft_x + shaft_w - 5, car_y + car_h,
                fill=color, outline=""
            )

            arrow = {"UP": "\u25b2", "DOWN": "\u25bc", "IDLE": "\u25a0"}.get(e.direction.value, "")
            canvas.create_text(shaft_x + shaft_w - 18, car_y + 12, text=arrow, fill="#1a1a1a", font=FONT_BOLD)

            canvas.create_text(
                shaft_x + 18, car_y + 12, text=f"F{e.current_floor}",
                fill="#1a1a1a", font=("Consolas", 9, "bold")
            )

            cols = 4
            head_r = 6
            pad_x, pad_y = 14, 28
            for p in range(e.current_load):
                col = p % cols
                row = p // cols
                cx = shaft_x + pad_x + col * (head_r * 2 + 4)
                cy = car_y + pad_y + row * (head_r * 2 + 4)
                canvas.create_oval(
                    cx - head_r, cy - head_r, cx + head_r, cy + head_r,
                    fill="#1a1a1a", outline=""
                )

        period = self.env.building.current_period.value
        completed = self.env.building.completed_count
        total_wait_time = sum(r.passenger.waiting_time for r in self.env.building.completed_requests)
        avg_wait = total_wait_time / completed if completed else 0
        self.info_var.set(
            f"Step {self.step_count}/{SIMULATION_STEPS}   Period: {period}   "
            f"Completed: {completed}   Waiting: {total_waiting(self.env.building)}   "
            f"Avg Wait: {avg_wait:.1f} steps   Reward: {self.total_reward:.1f}   "
            f"Policy: {self.strategy_var.get()}"
        )


# =================================================================
# Compare Strategies page
# =================================================================
class ComparePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG_DARK)
        self.app = app
        self.results = {}

        tk.Label(self, text="Compare Strategies", bg=BG_DARK, fg=FG_TEXT, font=FONT_HEADER).pack(
            anchor="w", padx=16, pady=(16, 4)
        )
        tk.Label(
            self,
            text="Runs each selected strategy silently on today's simulated traffic.",
            bg=BG_DARK, fg=FG_MUTED, font=FONT_NORMAL,
        ).pack(anchor="w", padx=16)

        check_frame = tk.Frame(self, bg=BG_DARK)
        check_frame.pack(anchor="w", padx=16, pady=10)
        self.check_vars = {}
        for name in STRATEGY_NAMES:
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(
                check_frame, text=name, variable=var, bg=BG_DARK, fg=FG_TEXT,
                selectcolor=BG_PANEL, font=FONT_NORMAL, activebackground=BG_DARK,
                activeforeground=FG_TEXT,
            ).pack(side="left", padx=8)
            self.check_vars[name] = var

        tk.Button(self, text="Run Comparison", command=self._run_comparison, font=FONT_BOLD).pack(
            anchor="w", padx=16, pady=(0, 8)
        )

        self.status_label = tk.Label(
            self, text="Ready — click Run Comparison to compare strategies on today's simulated traffic.",
            bg=BG_DARK, fg=FG_MUTED, font=FONT_NORMAL,
        )
        self.status_label.pack(anchor="w", padx=16)

        tk.Label(
            self, text=f"Note: 1 simulation step \u2248 {SECONDS_PER_STEP} seconds of real time.",
            bg=BG_DARK, fg=FG_MUTED, font=("Consolas", 9, "italic"),
        ).pack(anchor="w", padx=16, pady=(2, 8))

        columns = ("strategy", "completed", "waiting", "avg_wait", "reward")
        self.table = ttk.Treeview(self, columns=columns, show="headings", height=6)
        for col, label in zip(columns, ["Strategy", "Served", "Waiting", "Avg Wait (steps)", "Total Reward"]):
            self.table.heading(col, text=label)
            self.table.column(col, width=150, anchor="center")
        self.table.pack(fill="x", padx=16, pady=10)

        tk.Label(self, text="Average Waiting Time Comparison", bg=BG_DARK, fg=FG_TEXT, font=FONT_BOLD).pack(
            anchor="w", padx=16
        )
        self.chart = tk.Canvas(self, bg=BG_PANEL, height=300, highlightthickness=0)
        self.chart.pack(fill="both", expand=True, padx=16, pady=(4, 16))

    def _run_comparison(self):
        selected = [name for name, var in self.check_vars.items() if var.get()]
        if not selected:
            self.status_label.config(text="Select at least one strategy.")
            return

        self.status_label.config(text="Running...")
        self.update_idletasks()

        seed = self.app.current_seed

        self.results = {}
        for name in selected:
            self.results[name] = run_full_episode(name, seed=seed)

        self.status_label.config(
            text=f"Done — ran {len(selected)} strategies over {SIMULATION_STEPS} steps of today's simulated traffic."
        )
        self._update_table()
        self._draw_chart()

    def _update_table(self):
        for row in self.table.get_children():
            self.table.delete(row)
        for name, r in self.results.items():
            self.table.insert("", "end", values=(
                name, r["completed"], r["waiting"], f"{r['avg_wait']:.1f}", f"{r['reward']:.1f}"
            ))

    def _draw_chart(self):
        canvas = self.chart
        canvas.delete("all")
        width = canvas.winfo_width() or 700
        height = canvas.winfo_height() or 300
        if not self.results:
            return

        left_margin = 70
        bottom_margin = 60
        top_margin = 20
        plot_h = height - top_margin - bottom_margin
        plot_w = width - left_margin - 20

        max_wait = max(r["avg_wait"] for r in self.results.values()) or 1
        n = len(self.results)
        bar_w = min(100, plot_w / max(n, 1) - 20)

        canvas.create_line(left_margin, top_margin, left_margin, height - bottom_margin, fill=FG_MUTED)
        canvas.create_text(
            22, top_margin + plot_h / 2, text="Average Wait (steps)",
            fill=FG_MUTED, font=("Consolas", 10), angle=90, anchor="center"
        )

        canvas.create_line(left_margin, height - bottom_margin, width - 10, height - bottom_margin, fill=FG_MUTED)
        canvas.create_text(
            (left_margin + width) / 2, height - 14, text="Dispatch Strategy",
            fill=FG_MUTED, font=("Consolas", 10)
        )

        x = left_margin + 20
        for name, r in self.results.items():
            bar_h = (r["avg_wait"] / max_wait) * plot_h
            y0 = height - bottom_margin
            y1 = y0 - bar_h
            canvas.create_rectangle(x, y1, x + bar_w, y0, fill=COLOR_MOVING, outline="")
            canvas.create_text(x + bar_w / 2, y1 - 10, text=f"{r['avg_wait']:.1f}", fill=FG_TEXT, font=FONT_NORMAL)
            canvas.create_text(x + bar_w / 2, y0 + 15, text=name, fill=FG_MUTED, font=("Consolas", 9), width=bar_w + 10)
            x += bar_w + 20


# =================================================================
# Main app: sidebar + page container
# =================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Multi-Elevator Dispatch — Environment Visualizer")
        self.geometry("1150x720")
        self.configure(bg=BG_DARK)

        self.current_seed = random_seed_value()

        sidebar = tk.Frame(self, bg=BG_PANEL, width=180)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="Navigation", bg=BG_PANEL, fg=FG_TEXT, font=FONT_BOLD).pack(pady=(20, 10))
        tk.Button(sidebar, text="Live View", command=lambda: self._show("live"), width=16).pack(pady=6)
        tk.Button(sidebar, text="Compare Strategies", command=lambda: self._show("compare"), width=16).pack(pady=6)

        tk.Frame(sidebar, bg="#33363f", height=1).pack(fill="x", padx=16, pady=16)

        title_frame = tk.Frame(sidebar, bg=BG_PANEL)
        title_frame.pack(fill="both", expand=True)
        tk.Label(
            title_frame, text=SIDEBAR_TITLE, bg=BG_PANEL, fg=FG_TEXT,
            font=("Consolas", 15, "bold"), justify="center", wraplength=150,
        ).place(relx=0.5, rely=0.5, anchor="center")

        container = tk.Frame(self, bg=BG_DARK)
        container.pack(side="right", fill="both", expand=True)

        self.pages = {
            "live": LivePage(container, self),
            "compare": ComparePage(container, self),
        }
        for page in self.pages.values():
            page.place(relx=0, rely=0, relwidth=1, relheight=1)

        self._show("live")

    def _show(self, name):
        self.pages[name].tkraise()


if __name__ == "__main__":
    app = App()
    app.mainloop()