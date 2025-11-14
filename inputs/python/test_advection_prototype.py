import math
import pyquokka

prob_lo = 0.0
prob_hi = 1.0
nx = 4096
velocity = 1.0
max_time = 1.0
cfl = 0.4
err_tol = 0.007

def sawtooth(x: float, prob_lo: float, prob_hi: float) -> float:
    length = prob_hi - prob_lo
    shifted = math.fmod(x - prob_lo + length, length)
    return math.fmod(shifted + 0.5 * length, length)
  
def exact_func(grid: pyquokka.Grid, time: float) -> None:
    dx = grid.dx[0]
    ilo, ihi = grid.i_range
    length = prob_hi - prob_lo
    for i in range(ilo, ihi + 1):
        x = prob_lo + (float(i) + 0.5) * dx
        x0 = x - velocity * time
        while x0 < prob_lo:
            x0 += length
        while x0 >= prob_hi:
            x0 -= length
        grid.set_state(i, sawtooth(x0, prob_lo, prob_hi))

def init_func(grid: pyquokka.Grid) -> None:
    exact_func(grid, 0.0)
  
def diagnostics(time: float, err: float) -> None:
    print(f"Prototype time = {time}, L1(error) = {err}")

def main() -> int:
    pyquokka.initialize()
    sim = pyquokka.AdvectionSimulation()
    sim.configure_grid(nx, prob_lo, prob_hi)
    sim.set_advection_velocity(velocity, 0.0, 0.0)
    sim.set_cfl(cfl)
    sim.set_max_time(max_time)
    sim.set_max_timesteps(5000)
    sim.set_python_hooks(initial=init_func, exact=exact_func, diagnostics=diagnostics)
    sim.run()
    err = sim.error_norm()
    print(f"Prototype advection error norm = {err}")
    pyquokka.finalize()
    return 0 if err < err_tol else 1

if __name__ == "__main__":
    raise SystemExit(main())
