import matplotlib.pyplot as plt
import numpy as np
"""
This program calculates the position and velocity of a harmonic oscillator at time t given initial conditions.
"""

def x_analytic(t,x0,v0):
    """
    calculates the position of a harmonic oscillator at time t given initial conditions.
    """
    return x0 * np.cos(t) + v0 * np.sin(t)

def v_analytic(t,x0,v0):
    """
    calculates the velocity of a harmonic oscillator at time t given initial conditions.
    """
    return v0 * np.cos(t) - x0 * np.sin(t)

def x(dt,x0,v0):
    """
    calculates the position of a harmonic oscillator after a single timestep.
    """
    return x0 + v0 * dt + 0.5 * dt**2 * -x0 #F(0,x0,v0)

def v(dt,x0,v0,xt):
    """
    calculates the velocity of a harmonic oscillator after a timestep dt.
    The average force at the beginning and end of the timestep are used
    """
    return v0 + dt * (-x0 + -xt) / 2

def V(x):
    """
    Potential energy of a harmonic oscillator at position x.
    """
    return 0.5 * x**2

def T(v):
    """
    Kinetic energy of a harmonic oscillator at velocity v.
    """
    return 0.5 * v**2

def advance_time(total_time,dt,x0,v0):
    """
    Progressivly calculates the position and velocity of a harmonic oscillator after each timestep dt. Compares to the analytic solution.
    """
    v_array = [v0]
    x_array = [x0]
    PE_array = [V(x0)]
    KE_array = [T(v0)]
    Total_energy = [PE_array[0]+KE_array[0]]
    x_differences = []
    v_differences = []
    time_elapsed = 0

    x0 = Initial_x
    v0 = Initial_v
    while time_elapsed < total_time:
        xt = x(dt,x0,v0)
        vt = v(dt,x0,v0,xt)
        PE = V(xt)
        KE = T(vt)

        analitic_x = x_analytic(time_elapsed,Initial_x,Initial_v)
        analitic_v = v_analytic(time_elapsed,Initial_x,Initial_v)
        
        x_differences.append(xt-analitic_x)
        v_differences.append(vt-analitic_v)
       
        v_array.append(vt)
        x_array.append(xt)
        PE_array.append(PE)
        KE_array.append(KE)
        Total_energy.append(PE+KE)

        #update the variables for the next timestep
        x0 = xt
        v0 = vt
        time_elapsed += dt
        #print(f"At time t = {time_elapsed:.2f} the position is {xt:.3f}, the velocity is {vt:.5f} and the total energy is {PE+KE:.3f}")
   
    if total_time == 0:
        xt = x0
        vt = v0

    print(f"At time t = {total_time:.2f} the position is {xt:.3f} and the velocity is {vt:.5f}")
    print(f"The analytic position is {analitic_x:.3f} and the analytic velocity is {analitic_v:.5f}")
    return v_array, x_array, PE_array, KE_array, Total_energy, x_differences, v_differences

##############################################################################################################
#set initial conditions
Initial_x = 1
Initial_v = 0
Time_elapsed = 100
dt = 0.1

v_array, x_array, PE_array, KE_array, Total_energy, x_differences, v_differences = advance_time(Time_elapsed,dt,Initial_x,Initial_v)

#plot the position vs velocity of the harmonic oscillator and PE, KE and total energy vs time on different subplots. Show the differences of x and v vs time on their own sub plots
fig, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(x_array,v_array)
ax1.set_xlabel("Position")
ax1.set_ylabel("Velocity")
ax2.plot(PE_array, label = "Potential Energy")
ax2.plot(KE_array, label = "Kinetic Energy")
ax2.plot(Total_energy, label = "Total Energy")
ax2.set_xlabel("Time")
ax2.set_ylabel("Energy")
ax2.legend()
ax3.plot(x_differences, label = "X error")
ax3.plot(v_differences, label = "V error")
ax3.set_xlabel("Time")
ax3.set_ylabel("Errors")
ax3.legend()
plt.show()

