import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


class AnimatedDiffusion():
    def __init__(self, U, ind_excluded, N_x, N_r, del_x, del_r, del_t, u_max, coord_fixed, n_levels):

        N_tot = N_x * N_r ** 2
        N_pt = U.shape[1]
        U_full = np.zeros((N_tot, N_pt))
        mask = np.ones(N_tot, dtype=bool)
        mask[np.array(ind_excluded)] = False
        U_full[mask, :] = U
        U_full[np.logical_not(mask), :] = np.nan

        self.U = U_full
        self.ind_excluded = ind_excluded
        self.N_x = N_x
        self.N_r = N_r
        self.N_tot = N_tot
        self.N_pt = N_pt
        self.del_t = del_t
        self.del_x = del_x * 10 ** 4
        self.del_r = del_r * 10 ** 4
        self.u_max = u_max
        self.coord_fixed = coord_fixed
        self.n_levels = n_levels

        ind_fixed = np.zeros(3, dtype=bool)
        for ind, coord in enumerate(['i', 'j', 'k']):
            if list(coord_fixed.keys())[0] == coord:
                ind_fixed[ind] = True
        self.ind_varied = np.logical_not(ind_fixed)
        self.ind_fixed = ind_fixed
        self.scale_global = np.array([1, N_x, N_x*N_r])
        self.fixed_value = list(coord_fixed.values())[0]

        if list(coord_fixed.keys())[0] == 'i':
            self.N_max_1 = N_r
            self.del_cr1 = self.del_r
        else:
            self.N_max_1 = N_x
            self.del_cr1 = self.del_x

        # Set up the figure and axes
        self.fig = plt.figure(figsize=(5, 3), dpi=300)
        self.ax = self.fig.add_subplot()

        u_section = np.zeros((self.N_max_1, self.N_r))
        i_var = np.zeros(3)
        for c1 in range(self.N_max_1):
            for c2 in range(self.N_r):
                i_var[self.ind_varied] = np.array([c1, c2])
                q = self.fixed_value * np.dot(self.ind_fixed, self.scale_global) + np.dot(i_var, self.scale_global)
                u_section[c1, c2] = self.U[int(q), 0]

        self.cr1, self.cr2 = np.meshgrid(np.arange(self.N_max_1), np.arange(self.N_r))
        if list(coord_fixed.keys())[0] == 'i':
            self.coord1 = (self.cr1 - int(self.N_r / 2) + 1 / 2) * self.del_r
        else:
            self.coord1 = self.cr1 * self.del_cr1
        self.coord2 = (self.cr2 - int(self.N_r / 2) + 1 / 2) * self.del_r

        self.cf = self.ax.contourf(self.coord1, self.coord2, u_section.transpose(),
                                   cmap=plt.cm.get_cmap('jet'), vmin=0, vmax=u_max, levels=self.n_levels)
        self.cmap = plt.get_cmap("tab10")
        self.fig.colorbar(self.cf, ticks=[i/10 for i in range(11)])
        self.ax.set_title('$t={:.0f}$'.format(0*self.del_t) + ' [sec]')

        # Set plot range
        self.ax.set_aspect('equal')

        self.labels = ['x', 'y', 'z']
        self.lind_varied = np.arange(3)[self.ind_varied]

        self.ax.set_xlabel('$' + self.labels[self.lind_varied[0]] + '(\mu \mathrm{m})$')
        self.ax.set_ylabel('$' + self.labels[self.lind_varied[1]] + '(\mu \mathrm{m})$')
        self.fig.tight_layout()

        # Set up FuncAnimation
        self.ani = animation.FuncAnimation(self.fig, self.update, frames=self.N_pt, interval=self.N_pt)

    def update(self, t: int):

        u_section = np.zeros((self.N_max_1, self.N_r))
        i_var = np.zeros(3)
        for c1 in range(self.N_max_1):
            for c2 in range(self.N_r):
                i_var[self.ind_varied] = np.array([c1, c2])
                q = self.fixed_value * np.dot(self.ind_fixed, self.scale_global) + np.dot(i_var, self.scale_global)
                u_section[c1, c2] = self.U[int(q), t]

        # Update coordinate data in PathCollection
        self.ax.clear()
        self.cf = self.ax.contourf(self.coord1, self.coord2, u_section.transpose(), cmap=plt.cm.get_cmap('jet'),
                                   vmin=0, vmax=self.u_max, levels=self.n_levels)
        # Update title
        self.ax.set_title('$t={:.2f}$'.format(t*self.del_t/60/60) + ' [hour]')
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('$' + self.labels[self.lind_varied[0]] + '(\mu \mathrm{m})$')
        self.ax.set_ylabel('$' + self.labels[self.lind_varied[1]] + '(\mu \mathrm{m})$')

        return self.cf
