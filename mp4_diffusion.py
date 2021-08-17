import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import pickle as pkl
import os
from animate_diffusion import AnimatedDiffusion


def load_data(file_name):
    """
    :param file_name: file directory and name of pickle file to be loaded
    :return: variable contained by the pickle file
    """
    file_handle = open(file_name + '.pkl', 'rb')
    data = pkl.load(file_handle)
    file_handle.close()
    return data


file_name = 'diffusion_f1_result'

U, ind_excluded, N_x, N_r, del_x, del_r, del_t, u_max = load_data(file_name)

coord_fixed = {'j': int(N_r/2)}


def main():
    spc = AnimatedDiffusion(U, ind_excluded, N_x, N_r, del_x, del_r, del_t, u_max, coord_fixed, 50)
    anim = spc.ani

    # Save animation as gif
    # writergif = animation.PillowWriter(fps=6)
    # anim.save(file_name[:-6] + 'fixed_' + list(coord_fixed.keys())[0] + '_' + str(list(coord_fixed.values())[0])
    #           + '.gif', writer=writergif)
    writermp4 = animation.FFMpegWriter(fps=6)
    anim.save(file_name[:-6] + 'fixed_' + list(coord_fixed.keys())[0] + '_' + str(list(coord_fixed.values())[0])
              + '.mp4', writer=writermp4)

    plt.close('all')


if __name__ == "__main__":
    main()
