import lava.lib.dl.slayer as slayer
from torch import tensor
from matplotlib import animation
import matplotlib.pyplot as plt


def slayer_gif(array, filename, fps=24, dpi=300, figsize=(10,5)):
    height = array.shape[0]
    event = slayer.io.tensor_to_event(tensor(array).cpu().data.numpy().reshape(1, height ,-1))
    inp_anim = event.anim(plt.figure(figsize=figsize))
    inp_anim.save(f'{filename}.gif', animation.PillowWriter(fps=fps), dpi=dpi)

def numpy2event(array):
    height = array.shape[0]
    event = slayer.io.tensor_to_event(tensor(array).cpu().data.numpy().reshape(1, height ,-1))

    return event