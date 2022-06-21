import pygame as pg
import numpy as np
import quaternion as quat

pg.init()

white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 225)
red = (255, 0, 0)
black = (0, 0, 0)
c = np.zeros((8,3), dtype=object)
c[:4,:] = blue
c[4:,:] = red
Title_font = pg.font.SysFont('agencyfb', 30)
Time_font = pg.font.SysFont('agencyfb', 30)

file = "Quat_NDI_sim_0.csv"
y = np.genfromtxt(file, delimiter=",")
# y = np.genfromtxt("Euler_PD_sim.csv", delimiter=",")
t = y[0]
dt = np.roll(y[0], -1) - y[0]
dt[-1] = np.inf
q_sim = quat.from_float_array(np.roll(y[1:], 1, axis=0).T)
# q_sim = quat.from_euler_angles(np.roll(y[1:], 1, axis=0).T)

s =  800
window = pg.display.set_mode( (s, s) )
clock = pg.time.Clock()
cube_init = np.array([[-1, -1,  1],
                      [ 1, -1,  1],
                      [ 1,  1,  1],
                      [-1,  1,  1],
                      [-1, -1, -1],
                      [ 1, -1, -1],
                      [ 1,  1, -1],
                      [-1,  1, -1]])

def paused(pause):

    while pause:
        for event in pg.event.get():

            if event.type == pg.QUIT:
                print("Simulation Finished")
                pg.quit()
                quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_p:
                    pause = False
        pg.display.update()
        clock.tick(15)
lim_finish = np.array([106, 510, 908])
lim_start = np.array([98, 495, 896])
# lim_finish = np.array([280, 700, 1200])
# lim_start = np.array([94, 490, 890])
idx = [(x > lim_start[0] and x < lim_finish[0]) or
       (x > lim_start[1] and x < lim_finish[1]) or
       (x > lim_start[2] and x < lim_finish[2]) for x in t]
time_scale = 1

axis_init = np.identity(3)
scale = 100
d = 1000
displace = s/2
origin  = (displace, displace)
t = t[idx]
q_sim = q_sim[idx]
Title = Title_font.render(f'S/C Controller Model: {file[:-4]}', False, white)
for i,q in enumerate(q_sim):
    window.fill(black)
    Time_display = Time_font.render(f'T+{np.round(t[i])} [s]', False, white)
    print(np.round(t[i]))
    if t[i] > lim_start[0] and t[i] < lim_finish[0]:
        clock.tick(1 / dt[i]*time_scale)
    elif t[i] > lim_start[1] and t[i] < lim_finish[1]:
        clock.tick(1 / dt[i]*time_scale)
    elif t[i] > lim_start[2] and t[i] < lim_finish[2]:
        clock.tick(1 / dt[i]*time_scale)


    window.blit(Time_display, (s*0.05, s*0.95))
    window.blit(Title, (s*0.25, 0))
    cube = quat.rotate_vectors(q, cube_init, axis=-1)*scale
    axis = quat.rotate_vectors(q, axis_init, axis=-1)*scale*2/3
    x = (cube[:,0]) * d/(cube[:,-1]+d) + displace
    y = (cube[:,1]) * d/(cube[:,-1]+d) + displace
    x_ax = (axis[:,0]) * d/(axis[:,-1]+d) + displace
    y_ax = (axis[:,1]) * d/(axis[:,-1]+d) + displace

    x_ax, y_ax, z_ax = (x_ax[0], y_ax[0]), (x_ax[1], y_ax[1]), (x_ax[2], y_ax[2])


    front = list((map(tuple, np.vstack((x[:4], y[:4])).T)))
    back =  list((map(tuple, np.vstack((x[4:], y[4:])).T)))
    Rside = list((map(tuple, np.vstack((x[[2, 6, 5, 1]], y[[2, 6, 5, 1]])).T)))
    Lside = list((map(tuple, np.vstack((x[[3, 7, 4, 0]], y[[3, 7, 4, 0]])).T)))

    for i,xi in enumerate(x[:4]):
        pg.draw.circle(window, c[i], (xi, y[i]), 5)

    pg.draw.lines(window, white, True, front, 4)
    pg.draw.line(window, blue, origin, z_ax, 4)
    pg.draw.line(window, green, origin, y_ax, 4)
    pg.draw.line(window, red, origin, x_ax, 4)
    pg.draw.lines(window, white, False, Lside, 4)
    pg.draw.lines(window, white, False, Rside, 4)
    pg.draw.lines(window, red, True, back, 4)

    for i,xi in enumerate(x[4:]):
        pg.draw.circle(window, c[i+4], (xi, y[i+4]), 5)

    for event in pg.event.get():
        if event.type == pg.QUIT:
            pg.quit()
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_p:
                pause = True
                paused(pause)

    pg.display.update()
pause = True
paused(pause)
