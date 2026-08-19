import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

data = np.load("data/00000.npz")

frames = data["observations"]
actions = data["actions"]
rewards = data["rewards"]
lives = data["lives"]

num_frames = len(frames)


fig, (ax_img, ax_info) = plt.subplots(
    1, 2,
    figsize=(10, 5),
)

plt.subplots_adjust(bottom=0.18)


# ---- Image display ----
img = ax_img.imshow(frames[0])
ax_img.axis("off")
ax_img.set_title("Frame 0")


# ---- Information panel ----
ax_info.axis("off")

info_text = ax_info.text(
    0.05,
    0.8,
    "",
    fontsize=12,
    family="monospace",
    verticalalignment="top",
    transform=ax_info.transAxes,
)


def update(frame_idx):
    idx = int(frame_idx)

    img.set_data(frames[idx])

    ax_img.set_title(f"Frame {idx}")

    info_text.set_text(
        f"""
Frame: {idx}/{num_frames-1}
Action: {actions[idx]}
Reward: {rewards[idx]:.2f}
Lives: {lives[idx]}

Size: {frames[idx].shape}
""")

    fig.canvas.draw_idle()


# ---- Slider ----
slider_ax = plt.axes(
    [0.2, 0.05, 0.6, 0.03]
)

slider = Slider(
    slider_ax,
    "Frame",
    0,
    num_frames-1,
    valinit=0,
    valstep=1,
)

slider.on_changed(update)


update(0)

plt.show()