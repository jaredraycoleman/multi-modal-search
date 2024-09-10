from functools import lru_cache
from typing import Callable, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

N_STEPS = 10

def line_del(s: float, x: float, y: float, m: float, t: float) -> float:
    """Time it takes for a helper at (x,y) that has been moving toward (m,0)
    for time t to reach the robot at (s+t,0) and finish the delivery to (1,0)

    Args:
        s (float): starting position of robot
        x (float): x-coordinate of helper
        y (float): y-coordinate of helper
        m (float): x-coordinate of of helper focus point (m,0) - helper moves toward this point
            until the robot fails, at which point it moves toward the robot to pick up the object
            and complete the delivery to (1,0)
        t (float): fail time of the robot - thus the fail point of the robot is (s+t,0)
    """
    d2m = np.sqrt((m-x)**2 + y**2)  # distance from helper at (x,y) to (m,0)

    if s+t >= 1:
        return 1-s # robot fails after delivery is complete

    if t < d2m: # robot fails before helper reaches m
        a = t*(m-x)/d2m     # x-component of distance traveled by helper before robot fails
        b = t*y/d2m         # y-component of distance traveled by helper before robot fails
        delay = np.sqrt(((x+a)-(s+t))**2 + (y-b)**2) # distance between robot and helper when robot fails
        return (1-s) + delay # time taken to finish delivery to (1,0) after robot fails
    
    meeting = ((m-s)+d2m)/2         # time at which robot and helper would meet on the line
    if t < meeting or m-s < d2m: # helper does not reach robot before it fails
        return (d2m +  # take for helper to reach m
                np.abs(m-(s+t)) + # time for helper to reach robot from (m,0)
                1-(s+t)) # time taken to finish delivery to (1,0) after helper reaches robot
    else:
        return 1-s
    
def line_opt(s: float, x: float, y: float, t: float) -> float:
    """Optimal time for a helper at (x,y) to reach the robot at (s+t,0) and finish the delivery to (1,0)
    
    Args:
        s (float): starting position of robot
        x (float): x-coordinate of helper
        y (float): y-coordinate of helper
        t (float): time at which the robot fails
    """
    if s+t >= 1:
        return 1-s
    d2fail = np.sqrt((x-(s+t))**2 + y**2)
    return max(t, d2fail) + 1 - (s+t)
    
def best_line_cr(s: float, x: float, y: float, opt: Callable[[float], float]) -> Tuple[float, float, float]:
    """Find the best line for a helper at (x,y) to help the robot initially at (s,0) deliver to (1,0)

    Args:
        s (float): starting position of robot (also how much time has passed before helper starts moving)
        x (float): x-coordinate of helper
        y (float): y-coordinate of helper
        opt (Callable[[float], float]): optimal time function for helper to reach robot at (s,0) and finish delivery to (1,0)    

    Returns:
        Tuple[float, float, float]: Tuple of best m, best m worst t, and best cr such
            that the helper moves toward (m,0) until the robot fails at the worst-case
            time t, resulting in a competitive ratio of cr
    """
    best_m, best_m_worst_t, best_cr = 0, 0, np.inf
    for m in np.linspace(s, 1, N_STEPS):
        worst_t, worst_cr = 0, 1
        for t in np.linspace(0, 1-s, N_STEPS):
            cr = (s+line_del(s, x, y, m, t)) / opt(s+t)
            if cr > worst_cr:
                worst_t, worst_cr = t, cr
        if worst_cr < best_cr:
            best_m, best_m_worst_t, best_cr = m, worst_t, worst_cr
    return best_m, best_m_worst_t, best_cr

@lru_cache(maxsize=None)
def best_recurse_cr(s: float, x: float, y: float, num_turns: int, opt: Callable[[float], float]) -> Tuple[List[Tuple[float, float]], float, float]:
    if num_turns <= 1:
        m, m_worst_t, cr = best_line_cr(s, x, y, opt)
        return [(m, 0)], m_worst_t, cr
    if s >= 1: # if the robot fails after delivery, the CR is 1 (optimal)
        return [(1, 0)], 0, 1

    new_y = y - y/num_turns # y-coordinate of point helper moves toward
    best_new_x_worst_t, best_new_x_points, best_cr = 0, [], np.inf
    for m in np.linspace(s, 1, N_STEPS):
        new_x: float = x + ((m-x)/num_turns) # x-coordinate of point helper moves toward (given it is moving toward (m,0))
        d2new: float = np.sqrt((x-new_x)**2 + (y-new_y)**2)

        # if robot fails after helper reaches new_x, recurse
        new_s = min(1.0, s + d2new) # new starting point for robot (it has traveled while helper moved to new_x)
        _points, _new_x_worst_t, worst_cr = best_recurse_cr(new_s, new_x, new_y, num_turns-1, opt=opt)
        points = [(new_x, new_y)] + _points
        new_x_worst_t = d2new + _new_x_worst_t

        # get worst-case cr if robot fails before helper reaches new_x
        for t in np.linspace(0, d2new, N_STEPS): # robot fails before helper reaches new_x
            cr = (s+line_del(s, x, y, m, t)) / opt(s+t)
            if cr > worst_cr:
                new_x_worst_t = t
                worst_cr = cr

        if worst_cr < best_cr:
            best_new_x_worst_t, best_new_x_points, best_cr = new_x_worst_t, points, worst_cr

    return best_new_x_points, best_new_x_worst_t, best_cr


def example_1(x: float, y: float):
    best_m, best_m_worst_t, best_cr = best_line_cr(0, x, y, opt=lambda t: line_opt(0, x, y, t))

    plt.plot([0, 1], [0, 0], color='black')
    plt.plot([x, best_m], [y, 0], color='red')
    plt.scatter([0, best_m_worst_t], [0, 0], color='blue')
    a = best_m_worst_t*(best_m-x)/np.sqrt((best_m-x)**2 + y**2)
    b = best_m_worst_t*y/np.sqrt((best_m-x)**2 + y**2)
    plt.scatter([x+a], [y-b], color='green')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("example_1.png")
    plt.close()

def example_different_starting_point(x: float, y: float, s: float):
    m, m_worst_t, cr = best_line_cr(s, x, y, opt=lambda t: line_opt(s, x, y, t))

    plt.plot([0, 1], [0, 0], color='black')
    plt.plot([x, m], [y, 0], color='red')
    plt.scatter([s, s+m_worst_t], [0, 0], color='blue')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("example_different_starting_point.png")
    plt.close()

def example_recursive(x: float, y: float, s: float, num_turns: int):
    points, best_x_worst_t, best_cr = best_recurse_cr(s, x, y, num_turns, opt=lambda t: line_opt(s, x, y, t))

    print(f"Best x worst t: {best_x_worst_t}")
    print(f"Best cr: {best_cr}")

    _points = [(x, y)] + points
    cumsum = 0
    for i in range(1, len(_points)):
        cumsum += np.linalg.norm(np.array(_points[i]) - np.array(_points[i-1]))
        print(f"{i}: {cumsum}")

    plt.plot([0, 1], [0, 0], color='black')
    plt.scatter([x], [y], color='black')
    for i in range(len(points)-1):
        plt.scatter([points[i][0], points[i+1][0]], [points[i][1], points[i+1][1]], color='red')
    
    plt.scatter([s, s+best_x_worst_t], [0, 0], color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig("example_recursive.png")
    plt.close()

def example_4():
    y = 1/2

    xs = [0, 1/4, 1/2, 3/4, 1]
    num_turnses = [1, 2, 4, 6]
    
    # make len(xs) x len(num_turnses) grid of plots
    fig, axs = plt.subplots(len(xs), len(num_turnses), figsize=(5*len(num_turnses), 3*len(xs)))
    progress, total = 0, len(xs)*len(num_turnses)
    for i, x in enumerate(xs):
        for j, num_turns in enumerate(num_turnses):
            progress += 1
            print(f"Progress: {progress}/{total}")
            points, best_x_worst_t, best_cr = best_recurse_cr(0, x, y, num_turns, opt=lambda t: line_opt(0, x, y, t))

            axs[i, j].plot([0, 1], [0, 0], color='black')
            axs[i, j].scatter([x], [y], color='black')
            for point in points:
                axs[i, j].scatter([point[0]], [point[1]], color='red')
            
            axs[i, j].scatter([best_x_worst_t], [0], color='blue')
            axs[i, j].set_aspect('equal', adjustable='box')
            axs[i, j].set_title(f"x={x}, num_turns={num_turns}, CR={best_cr:.2f}")

    plt.savefig("example_4.png")
    plt.close()


def example_5(num_turns: int):
    # same as example four but with fixed num_turns=5 and varying x and y
    xs = [0, 1/4, 1/2, 3/4, 1]
    ys = [1/4, 1/2, 3/4, 1]

    # make len(xs) x len(ys) grid of plots
    fig, axs = plt.subplots(len(xs), len(ys), figsize=(5*len(ys), 5*len(xs)))
    progress, total = 0, len(xs)*len(ys)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            progress += 1
            print(f"Progress: {progress}/{total}")
            points, best_x_worst_t, best_cr = best_recurse_cr(0, x, y, num_turns, opt=lambda t: line_opt(0, x, y, t))

            axs[i, j].plot([0, 1], [0, 0], color='black')
            axs[i, j].scatter([x], [y], color='black')
            for point in points:
                axs[i, j].scatter([point[0]], [point[1]], color='red')
            
            axs[i, j].scatter([best_x_worst_t], [0], color='blue')
            axs[i, j].set_aspect('equal', adjustable='box')
            axs[i, j].set_title(f"({x}, {y}), turns={num_turns}, CR={best_cr:.4f}")

            # set y range to (0, 1) for all plots
            axs[i, j].set_ylim(0, 1)

    plt.savefig(f"example_{num_turns}_turns.png")
    plt.close()

def main():
    # s, x, y, num_turns = 0, 1/2, 1/2, 6
    # example_1(x, y)
    # example_different_starting_point(x, y, s)
    # example_recursive(x, y, s, num_turns)

    # example_4()
    example_5(num_turns=1)
    example_5(num_turns=2)
    example_5(num_turns=5)

if __name__ == "__main__":
    main()
