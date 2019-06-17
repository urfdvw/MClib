import functions as fn
import numpy as np

N = 2000
maxx = 10
minx = -10

x = np.linspace(minx, maxx, N)
w = np.zeros([N, N])
y = np.zeros([N*N, 2])

iy = 0
for i in range(N):
    for j in range(N):
        x_now = np.zeros([1, 2])
        x_now[0, 0] = x[i]
        x_now[0, 1] = x[j]
        y[iy, :] = x_now
        iy += 1

logw = fn.TwoDlogbanana(y)
w = fn.logw2w(logw)

print(fn.weightedsum(y, w))