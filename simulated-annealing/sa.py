import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt
import matplotlib as mpl
import math

mpl.rcParams['figure.figsize'] = (19, 8)

def annealing(maxsteps=100,annealing_type='fast',t_start=1,print_info=False):
    x_var = gen_start_point()
    y_var = objective(x_var)
    history_x_vars, history_y_vars = [x_var], [y_var]
    t_current=t_start
    for step in range(1,maxsteps):
        fraction = step / float(maxsteps)
        new_x_var = random_neighbour(x_var, fraction,t_current)
        new_y_var = objective(new_x_var)
        delta = new_y_var - y_var
        accept=make_choice(delta,t_current)

        if accept:
            x_var, y_var = new_x_var, new_y_var
            history_x_vars.append(x_var)
            history_y_vars.append(y_var)
            if print_info:
              print_data(step, x_var ,y_var)

        if annealing_type=='fast':
          t_current=temp_fast(step+1,t_start)
        elif annealing_type == 'boltz':
          t_current=temp_boltz(step+1,t_start)
        elif annealing_type == 'custom':
          t_current = temp_custom(fraction,t_start)
    return x_var, objective(x_var), history_x_vars, history_y_vars

def gen_start_point():
    # a, b = interval
    # return rd.randint(a,b)
    return 9


def make_choice(delta,t_current):
    accept=False
    if delta<0:
        accept=True
    else:
        p = np.exp(-delta / t_current)
        if p>rd.uniform():
            accept=True
    return  accept

def visualize(history_x_vars, history_y_vars):
    fig, axs = plt.subplots(3)
    axs[0].plot(history_x_vars, 'r')
    axs[0].set_title("Изменение значений x")
    axs[1].plot(history_y_vars, 'b')
    axs[1].set_title("Изменение значений целевой функции")
    a, b=interval
    params= np.arange(a, b, 0.01)
    res=[objective(x) for x in params]
    axs[2].plot(params, res,color='g')
    axs[2].scatter(history_x_vars[-1], history_y_vars[-1], s=30,color='b')
    axs[2].scatter(history_x_vars[0], history_y_vars[0], s=30,color='r')
    axs[2].set_title("Результат")

def print_data(step, x, y):
    print("Итерация: ",step)
    print("x = ",x, " y = ",y)

def objective(x):
    return x*x*(2+abs(math.sin(4*x)))

def bound(x):
    a, b = interval
    return min(max(x, a), b)

def random_neighbour(x,fraction,t):
    a, b = interval
    amplitude = (b - a) * fraction / 10
    delta = (-amplitude/2) + amplitude * rd.uniform()
    return bound(x + delta)
    # return bound(x+ rd.randint(-1,1)*t*rd.uniform(0, 1))


def temp_boltz(k,t_s):
  return t_s/math.log(k+1, math.e)

def temp_fast(k,t_s):
  return t_s/k

def temp_custom(fraction,t_s):
    return max(0.01, min(t_s, t_s - fraction))

# rd.seed(159)
interval = (-10, 10)
x_var, y_var, history_x_vars, history_y_vars = annealing(maxsteps=190,annealing_type='custom',t_start=1,print_info=False)
print("-"*11,"Результат","-"*11)
print("x = ",x_var)
print("Значение целевой функции = ",y_var)
visualize(history_x_vars, history_y_vars)