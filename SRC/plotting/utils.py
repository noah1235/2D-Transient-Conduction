import matplotlib as plt

def set_grid(ax):
    ax.set_axisbelow(True)
    ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5, direction="in")
    ax.tick_params(axis="y", which="minor", bottom=False, top=False, left=False, right=False)

def save_svg(mpl, fig, path):
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['svg.hashsalt'] = ''
    plt.rcParams['font.family'] = 'Arial'


    # optional: tighter bounding box & transparent background
    fig.savefig(
    path,
    bbox_inches="tight",
    pad_inches=0.02,
    transparent=True,
    metadata={"Title": "Nice vector plot", "Creator": "matplotlib"}
    )