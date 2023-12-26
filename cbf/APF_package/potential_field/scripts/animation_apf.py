#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LogNorm
from matplotlib import animation

import jax.numpy as jnp
from jax import grad, jit, vmap,value_and_grad
from scipy.optimize import minimize
from collections import defaultdict
from itertools import zip_longest
from functools import partial

def init():
    line.set_data([], [])
    line.set_3d_properties([])
    # point.set_data([], [])
    # point.set_3d_properties([])
    # ax.plot_surface(x_s+4.0, y_s+4.5, z_s+f(4.0,4.5)+r, color=np.random.choice(['g','b']), alpha=0.8)
    # sphere = ax.plot_surface(x_s+4.0, y_s+4.5, z_s+f(4.0,4.5)+r, color=np.random.choice(['g','b']), alpha=0.8)

    f_sphere = lambda x, y: r**2-(x-4.0)**2-(y-4.5)**2
    xmin, xmax, xstep = 3.6,4.4,0.01#3.95, 4.05, .01
    ymin, ymax, ystep = 4.1,4.9,0.01#4.45, 4.55, .01
    x_s, y_s = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z_s = f_sphere(x_s, y_s) + r
    ax.plot_surface(x_s, y_s, z_s, color=np.random.choice(['g','b']), alpha=0.8)    # plt.show()
    sphere = ax.plot_surface(x_s, y_s, z_s, color=np.random.choice(['g','b']), alpha=0.8)

    sphere._facecolors2d = sphere._facecolors3d
    sphere._edgecolors2d=sphere._edgecolors3d

    return line,sphere#, point

def animate(i):
    line.set_data(path[0,:i], path[1,:i])
    line.set_3d_properties(f(*path[::,:i]))
    # point.set_data(path[0,i-1:i], path[1,i-1:i])
    # point.set_3d_properties(f(*path[::,i-1:i]))
   
    # ax.plot_surface(x_s-path[0,i-1:i], y_s-path[1,i-1:i], z_s-f(*path[::,:i]), color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)
    # print("sdsd",path[0,i-1:i][0],len(path[0,i-1:i]))
    
    # ax.plot_surface(x_s-path[0,i-1:i], y_s-path[1,i-1:i], z_s-f(*path[::,:i]), color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)
    # sphere = ax.plot_surface(x_s-path[0,i-1:i], y_s-path[1,i-1:i], z_s-f(*path[::,:i]), color=np.random.choice(['g','b']), alpha=0.5*np.random.random()+0.5)
   
    # ax.plot_surface(x_s+path[0][i:i+1][0], y_s+path[1][i:i+1][0], z_s+f(path[0][i:i+1][0],path[1][i:i+1][0])+r, color=np.random.choice(['g','b']), alpha=0.8)
    # sphere = ax.plot_surface(x_s+path[0][i:i+1][0], y_s+path[1][i:i+1][0], z_s+f(path[0][i:i+1][0],path[1][i:i+1][0])+r, color=np.random.choice(['g','b']), alpha=0.8)    
    f_sp = lambda x, y: r**2-(x-path[0][i:i+1][0])**2-(y-path[1][i:i+1][0])**2
    xmin_, xmax_, xstep_ = path[0][i:i+1][0]-r, path[0][i:i+1][0]+r, .01
    ymin_, ymax_, ystep_ = path[1][i:i+1][0]-r, path[1][i:i+1][0]+r, .01
    x_sp, y_sp = np.meshgrid(np.arange(xmin_, xmax_ + xstep_, xstep_), np.arange(ymin_, ymax_ + ystep_, ystep_))
    z_sp = f_sp(x_sp, y_sp) + r
    ax.plot_surface(x_sp, y_sp, z_sp, color=np.random.choice(['g','b']), alpha=0.8)    # plt.show()
    # print(kk)
    # sphere = ax.plot_surface(x_s+4.0, y_s+4.5, z_s+f(4.0,4.5)+r, color=np.random.choice(['g','b']), alpha=0.8)
    sphere = ax.plot_surface(x_sp, y_sp, z_sp, color=np.random.choice(['g','b']), alpha=0.8)
    
    
    sphere._facecolors2d = sphere._facecolors3d
    sphere._edgecolors2d=sphere._edgecolors3d

    return line,sphere# , point

def make_minimize_cb(path=[]):
    
    def minimize_cb(xk):
        path.append(np.copy(xk))

    return minimize_cb

if __name__ == '__main__':

    f = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    xmin, xmax, xstep = 2.8, 4, .2
    ymin, ymax, ystep = -4.5, 4.5, .2
    x_list, y_list = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    z = f(x_list, y_list)

    # minima_ = np.array([2.19966526, 0.19753339])
    minima_ = np.array([3.08621227, 0.54958576])
    minima_ = minima_.reshape(-1, 1)

    fig = plt.figure(figsize=(8, 5))
    ax = plt.axes(projection='3d', elev=50, azim=-50)

    ax.plot_surface(x_list, y_list, z, norm=LogNorm(), rstride=1, cstride=1, edgecolor='none', alpha=.8, cmap=plt.cm.jet)
    ax.plot(*minima_, f(*minima_), 'r*', markersize=10)

    line, = ax.plot([], [], [], 'b', label='Gradient-Descent', lw=2)
    # point, = ax.plot([], [], [], 'bo')
    # u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]
    r=0.4
    # x_s = r*np.cos(u)*np.sin(v)
    # y_s = r*np.sin(u)*np.sin(v)
    # z_s = r*np.cos(v)
    
    # f_sphere = lambda x, y: r**2-(x-4.0)**2-(y-4.5)**2
    # xmin, xmax, xstep = 3.6,4.4,0.01#3.95, 4.05, .01
    # ymin, ymax, ystep = 4.1,4.9,0.01#4.45, 4.55, .01
    # x_s, y_s = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))
    # z_s = np.sqrt(f_sphere(x_s, y_s)) + r
    # ax.plot_surface(x_s, y_s, z_s, color=np.random.choice(['g','b']), alpha=0.8)    
    # plt.show()
    # print(kk)
    # sphere = ax.plot_surface(x_s, y_s, z_s, color=np.random.choice(['g','b']), alpha=0.8)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$z$')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    
    x0 = np.array([4., 4.5])

    func = value_and_grad(lambda args: f(*args))
    path_ = [x0]
    res = minimize(func, x0=x0,method='Newton-CG',
               jac=True, tol=1e-20, callback=make_minimize_cb(path_))

    path = np.array(path_).T

    # anim = animation.FuncAnimation(fig, animate, init_func=init,
    #                            frames=path.shape[1], interval=200, 
    #                            repeat_delay=5, blit=True)

    plt.show()






