from scipy import sparse
from scipy.sparse import linalg
import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
from time import perf_counter


def save_data(data, file_name):
    """
    :param data: class instance ot be saved
    :param file_name: file directory and name for the saved pickle file
    :return: None
    """
    filehandler = open(file_name + '.pkl', 'wb')
    pkl.dump(data, filehandler)
    filehandler.close()


def ind_transformation(i, j, k):
    return i + j*N_x + k*N_x*N_r


# This script solves the diffusion reaction equation modeling the diffusible factor regulating symmetric stem cell
# division and stem cell activation.
# The Forward (explicit) Euler finite difference scheme is used due to the high computational demand.

# PARAMETERS -----------------------------------------------------------------------------------------------------------

# MODEL PARAMETERS

# decay rate in [1/sec]
lam = 0.0005
# diffusion coefficient in [cm^2/sec]
alp = 3.3 * 10 ** (-8)
# constant concentration along central canal surface in [mol/cm^2]
u_c = 1
# cell size in [cm]
s_cell = 6 * 10 ** (-4)
# box sizes in [cell size]
R_max = 15
L_max = 50
# central canal sizes in [cell size]
R_c = 1.5
L_c = 45

# NUMERICAL SCHEME PARAMETERS

# number of steps per cell in radial direction
d_r = 1
# number of steps per cell in longitudinal direction
d_x = 1
# time length of simulation in [sec]
t_sim = 60 * 60 * 24
# time step safety factor against instability
sf = 1
# time step for saving data in [sec]
del_t_save = 60 * 10

# DERIVED PARAMETERS

# number of grid points in radial directions
N_r = (2 * R_max + 1) * d_r + 1
# number of grid points in longitudinal direction
N_x = L_max * d_x + 1
# total number of grid points
N_tot = N_x * N_r * N_r
# step sizes in [cm]
del_x = s_cell / d_x
del_r = s_cell / d_r
# time step at the boundary of stability for the diffusion equation (without reaction term) in [sec]
del_t_crit = 1 / (2 * alp * (1 / (del_x ** 2) + 2 / (del_r ** 2)))
# chosen time step in [sec]
del_t = del_t_crit / sf
# number of time steps
N_t = int(t_sim / del_t)
# Fourier numbers
F_x = alp * del_t / (del_x ** 2)
F_r = alp * del_t / (del_r ** 2)
# normalized decay rate
lam_star = lam * del_t

# printing chosen time step
print('chosen time step:', del_t, '[sec]')

# BUILDING COEFFICIENT MATRIX ------------------------------------------------------------------------------------------

t_start = perf_counter()

# Middle Region --------------------------------------------------------------------------------------------------------

row_qq, Aqq, col_xm, Aqxm, col_xp, Aqxp, col_ym, Aqym, col_yp, Aqyp, col_zm, Aqzm, col_zp, Aqzp = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(1, N_x-1):
    for j in range(1, N_r-1):
        for k in range(1, N_r-1):
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            col_xm.append(q - 1)
            col_xp.append(q + 1)
            col_ym.append(q - N_x)
            col_yp.append(q + N_x)
            col_zm.append(q - N_x * N_r)
            col_zp.append(q + N_x * N_r)
            Aqq.append(-2 * (F_x + 2 * F_r) - lam_star)
            Aqxm.append(F_x)
            Aqxp.append(F_x)
            Aqym.append(F_r)
            Aqyp.append(F_r)
            Aqzm.append(F_r)
            Aqzp.append(F_r)

row_mid = np.array(row_qq * 7)
col_mid = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_mid = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# Boundaries Along the Box ---------------------------------------------------------------------------------------------

# FACES

# along faces at fixed i
row_qq, Aqq, row_xm, col_xm, Aqxm, row_xp, col_xp, Aqxp, col_ym, Aqym, col_yp, Aqyp, col_zm, Aqzm, col_zp, Aqzp = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in [0, N_x-1]:
    for j in range(1, N_r-1):
        for k in range(1, N_r-1):
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if i == 0:
                row_xp.append(q)
                col_xp.append(q + 1)
                Aqxp.append(F_x)
            else:
                row_xm.append(q)
                col_xm.append(q - 1)
                Aqxm.append(F_x)
            col_ym.append(q - N_x)
            col_yp.append(q + N_x)
            col_zm.append(q - N_x*N_r)
            col_zp.append(q + N_x*N_r)
            Aqq.append(- F_x - 4 * F_r - lam_star)
            Aqym.append(F_r)
            Aqyp.append(F_r)
            Aqzm.append(F_r)
            Aqzp.append(F_r)

row_face_i = np.concatenate((row_qq, row_xm, row_xp, row_qq * 4))
col_face_i = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_face_i = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# along faces at fixed j
row_qq, Aqq, col_xm, Aqxm, col_xp, Aqxp, row_ym, col_ym, Aqym, row_yp, col_yp, Aqyp, col_zm, Aqzm, col_zp, Aqzp = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(1, N_x-1):
    for j in [0, N_r-1]:
        for k in range(1, N_r-1):
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if j == 0:
                row_yp.append(q)
                col_yp.append(q + N_x)
                Aqyp.append(F_r)
            else:
                row_ym.append(q)
                col_ym.append(q - N_x)
                Aqym.append(F_r)
            col_xm.append(q - 1)
            col_xp.append(q + 1)
            col_zm.append(q - N_x*N_r)
            col_zp.append(q + N_x*N_r)
            Aqq.append(- F_r - 2 * (F_x + F_r) - lam_star)
            Aqxm.append(F_x)
            Aqxp.append(F_x)
            Aqzm.append(F_r)
            Aqzp.append(F_r)

row_face_j = np.concatenate((row_qq * 3, row_ym, row_yp, row_qq * 2))
col_face_j = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_face_j = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# along faces at fixed k
row_qq, Aqq, col_xm, Aqxm, col_xp, Aqxp, col_ym, Aqym, col_yp, Aqyp, row_zm, col_zm, Aqzm, row_zp, col_zp, Aqzp = \
    [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(1, N_x-1):
    for j in range(1, N_r-1):
        for k in [0, N_r-1]:
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if k == 0:
                row_zp.append(q)
                col_zp.append(q + N_x*N_r)
                Aqzp.append(F_r)
            else:
                row_zm.append(q)
                col_zm.append(q - N_x*N_r)
                Aqzm.append(F_r)
            col_xm.append(q - 1)
            col_xp.append(q + 1)
            col_ym.append(q - N_x)
            col_yp.append(q + N_x)
            Aqq.append(- F_r - 2 * (F_x + F_r) - lam_star)
            Aqxm.append(F_x)
            Aqxp.append(F_x)
            Aqym.append(F_r)
            Aqyp.append(F_r)

row_face_k = np.concatenate((row_qq * 5, row_zm, row_zp))
col_face_k = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_face_k = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# assembling all face coordinates
row_face = np.concatenate((row_face_i, row_face_j, row_face_k))
col_face = np.concatenate((col_face_i, col_face_j, col_face_k))
A_face = np.concatenate((A_face_i, A_face_j, A_face_k))

# EDGES

# at fixed i and j values
row_qq, row_xm, row_xp, row_ym, row_yp, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp, \
Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp = \
[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in [0, N_x-1]:
    for j in [0, N_r-1]:
        for k in range(1, N_r-1):
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if i == 0:
                row_xm.append(q)
                col_xm.append(q + 1)
                Aqxm.append(F_x)
            else:
                row_xp.append(q)
                col_xp.append(q - 1)
                Aqxp.append(F_x)
            if j == 0:
                row_ym.append(q)
                col_ym.append(q + N_x)
                Aqym.append(F_r)
            else:
                row_yp.append(q)
                col_yp.append(q - N_x)
                Aqyp.append(F_r)
            col_zm.append(q - N_x*N_r)
            col_zp.append(q + N_x*N_r)
            Aqq.append(- F_x - 3 * F_r - lam_star)
            Aqzm.append(F_r)
            Aqzp.append(F_r)

row_edge_ij = np.concatenate((row_qq, row_xm, row_xp, row_ym, row_yp, row_qq * 2))
col_edge_ij = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_edge_ij = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# at fixed j and k values
row_qq, row_ym, row_yp, row_zm, row_zp, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp, \
Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp = \
[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in range(1, N_x-1):
    for j in [0, N_r-1]:
        for k in [0, N_r-1]:
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if j == 0:
                row_ym.append(q)
                col_ym.append(q + N_x)
                Aqym.append(F_r)
            else:
                row_yp.append(q)
                col_yp.append(q - N_x)
                Aqyp.append(F_r)
            if k == 0:
                row_zm.append(q)
                col_zm.append(q + N_x*N_r)
                Aqzm.append(F_r)
            else:
                row_zp.append(q)
                col_zp.append(q - N_x*N_r)
                Aqzp.append(F_r)
            col_xm.append(q - 1)
            col_xp.append(q + 1)
            Aqq.append(- 2 * (F_x + F_r) - lam_star)
            Aqxm.append(F_x)
            Aqxp.append(F_x)

row_edge_jk = np.concatenate((row_qq * 3, row_ym, row_yp, row_zm, row_zp))
col_edge_jk = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_edge_jk = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# at fixed i and k values
row_qq, row_xm, row_xp, row_zm, row_zp, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp, \
Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp = \
[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in [0, N_x-1]:
    for j in range(1, N_r-1):
        for k in [0, N_r-1]:
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if i == 0:
                row_xm.append(q)
                col_xm.append(q + 1)
                Aqxm.append(F_x)
            else:
                row_xp.append(q)
                col_xp.append(q - 1)
                Aqxp.append(F_x)
            if k == 0:
                row_zm.append(q)
                col_zm.append(q + N_x*N_r)
                Aqzm.append(F_r)
            else:
                row_zp.append(q)
                col_zp.append(q - N_x*N_r)
                Aqzp.append(F_r)
            col_ym.append(q - N_x)
            col_yp.append(q + N_x)
            Aqq.append(- F_x - 3 * F_r - lam_star)
            Aqym.append(F_r)
            Aqyp.append(F_r)

row_edge_ik = np.concatenate((row_qq, row_xm, row_xp, row_qq * 2, row_zm, row_zp))
col_edge_ik = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_edge_ik = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# assembling all edge coordinates
row_edge = np.concatenate((row_edge_ij, row_edge_jk, row_edge_ik))
col_edge = np.concatenate((col_edge_ij, col_edge_jk, col_edge_ik))
A_edge = np.concatenate((A_edge_ij, A_edge_jk, A_edge_ik))

# GRID POINTS

# at fixed i and k values
row_qq, row_xm, row_xp, row_ym, row_yp, row_zm, row_zp, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp, \
Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp = \
[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
for i in [0, N_x-1]:
    for j in [0, N_r-1]:
        for k in [0, N_r-1]:
            q = ind_transformation(i, j, k)
            row_qq.append(q)
            if i == 0:
                row_xm.append(q)
                col_xm.append(q + 1)
                Aqxm.append(F_x)
            else:
                row_xp.append(q)
                col_xp.append(q - 1)
                Aqxp.append(F_x)
            if j == 0:
                row_ym.append(q)
                col_ym.append(q + N_x)
                Aqym.append(F_r)
            else:
                row_yp.append(q)
                col_yp.append(q - N_x)
                Aqyp.append(F_r)
            if k == 0:
                row_zm.append(q)
                col_zm.append(q + N_x * N_r)
                Aqzm.append(F_r)
            else:
                row_zp.append(q)
                col_zp.append(q - N_x * N_r)
                Aqzp.append(F_r)
            Aqq.append(- F_x - 2 * F_r - lam_star)

# assembling all grid point coordinates
row_point = np.concatenate((row_qq, row_xm, row_xp, row_ym, row_yp, row_zm, row_zp))
col_point = np.concatenate((row_qq, col_xm, col_xp, col_ym, col_yp, col_zm, col_zp))
A_point = np.concatenate((Aqq, Aqxm, Aqxp, Aqym, Aqyp, Aqzm, Aqzp))

# assembling all rows and columns
rows = np.concatenate((row_mid, row_face, row_edge, row_point))
cols = np.concatenate((col_mid, col_face, col_edge, col_point))
A_vals = np.concatenate((A_mid, A_face, A_edge, A_point))

# Boundaries Along the Central Canal -----------------------------------------------------------------------------------

row_cc = []
# at fixed i values
for i in [L_c * d_x]:
    for j in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
        for k in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
            q = ind_transformation(i, j, k)
            row_cc.append(q)

# at fixed j values
for i in range(d_x * L_c):
    for j in [d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1)]:
        for k in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
            q = ind_transformation(i, j, k)
            row_cc.append(q)

# at fixed k values
for i in range(d_x * L_c):
    for j in range(d_r * (R_max - int(R_c)) + 1, d_r * (R_max + int(R_c) + 1)):
        for k in [d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1)]:
            q = ind_transformation(i, j, k)
            row_cc.append(q)

# Excluding Redundant Matrix Entries -----------------------------------------------------------------------------------

ind_redundant = np.zeros_like(rows, dtype=bool)
for q in row_cc:
    ind_redundant_q = rows == q
    ind_redundant = np.logical_or(ind_redundant_q, ind_redundant)

ind_kept = np.logical_not(ind_redundant)
rows_final = rows[ind_kept]
cols_final = cols[ind_kept]
A_final = A_vals[ind_kept]

t_end = perf_counter()

print('time requirement for construction of sparse matrix of size (' + str(N_tot) + ', ' + str(N_tot) + '):',
      t_end - t_start, '[sec]')

A = sparse.coo_matrix((A_final, (rows_final, cols_final)), shape=(N_tot, N_tot))
I = sparse.coo_matrix((np.ones(N_tot), (np.arange(N_tot), np.arange(N_tot))), shape=(N_tot, N_tot))

print('largest-magnitude eigenvalue:', linalg.eigs(A + I, k=1)[0][0])

# INITIALIZATION -------------------------------------------------------------------------------------------------------

u = np.zeros(N_tot)
# at fixed i values
for i in [L_c * d_x]:
    for j in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
        for k in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
            q = ind_transformation(i, j, k)
            u[q] = u_c

# at fixed j values
for i in range(d_x * L_c):
    for j in [d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1)]:
        for k in range(d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1) + 1):
            q = ind_transformation(i, j, k)
            u[q] = u_c

# at fixed k values
for i in range(d_x * L_c):
    for j in range(d_r * (R_max - int(R_c)) + 1, d_r * (R_max + int(R_c) + 1)):
        for k in [d_r * (R_max - int(R_c)), d_r * (R_max + int(R_c) + 1)]:
            q = ind_transformation(i, j, k)
            u[q] = u_c


# ELIMINATION OF GRID POINTS INSIDE CENTRAL CANAL  ---------------------------------------------------------------------

# collecting indices that are inside the central canal
ind_excluded = []
for i in range(d_x * L_c):
    for j in range(d_r * (R_max - int(R_c)) + 1, d_r * (R_max + int(R_c) + 1)):
        for k in range(d_r * (R_max - int(R_c)) + 1, d_r * (R_max + int(R_c) + 1)):
            ind_excluded.append(ind_transformation(i, j, k))

A_csr = A.tocsr(copy=True)
mask = np.ones(N_tot, dtype=bool)
mask[ind_excluded] = False
A_trunc_csr = A_csr[mask][:, mask]
u = u[mask]

# A_trunc_coo = A_trunc_csr.tocoo(copy=True)
# A_trunc_dia = A_trunc_csr.todia(copy=True)

# TIME ITERATION -------------------------------------------------------------------------------------------------------

# measuring computational time
t_start = perf_counter()
N_save = int(t_sim / del_t_save)
U_save = np.zeros((u.shape[0], N_save + 1))
U_save[:, 0] = u
U_old, count = u, 1
for t in range(N_t):
    U = U_old + A_trunc_csr.dot(U_old)
    t_save = count * del_t_save
    t_overhang = (t + 1) * del_t % t_save
    if 0 < t_overhang < del_t:
        U_save[:, count] = U_old * (1 - t_overhang / del_t) + U * (t_overhang / del_t)
        count += 1
    elif 0 == t_overhang:
        U_save[:, count] = U
        count += 1
    U_old = U
t_end = perf_counter()
print('time requirement of performing ' + str(N_t) + ' number of steps:', t_end - t_start, '[sec]')

save_data([U_save, ind_excluded, N_x, N_r, del_x, del_r, del_t_save, u_c], 'diffusion_f1_result')
