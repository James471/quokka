#include <cmath>
#include <utility>

#include "AMReX.H"
#include "AMReX_GpuLaunch.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "experimental/DescriptorPrototype.hpp"

using quokka::experimental::AdvectionSimulationPrototype;

namespace
{

struct ProblemConfig {
	amrex::Real prob_lo = 0.0;
	amrex::Real prob_hi = 1.0;
	amrex::Real advection_velocity = 0.0;
};

ProblemConfig g_problem{};

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE auto sawtooth(amrex::Real x, amrex::Real prob_lo, amrex::Real prob_hi) -> amrex::Real
{
	const amrex::Real length = prob_hi - prob_lo;
	amrex::Real shifted = x - prob_lo;
	shifted = std::fmod(shifted + length, length);
	return std::fmod(shifted + 0.5 * length, length);
}

void exactSolution(quokka::grid const &grid_elem, amrex::Real time)
{
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx = grid_elem.dx_;
	const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo = grid_elem.prob_lo_;
	const amrex::Box &indexRange = grid_elem.indexRange_;
	const amrex::Array4<double> &state = grid_elem.array_;
	const amrex::Real prob_lo_x = g_problem.prob_lo;
	const amrex::Real prob_hi_x = g_problem.prob_hi;
	const amrex::Real velocity = g_problem.advection_velocity;
	const amrex::Real length = prob_hi_x - prob_lo_x;

	amrex::ParallelFor(indexRange, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
		amrex::Real const x = prob_lo[0] + (static_cast<amrex::Real>(i) + 0.5) * dx[0];
		amrex::Real x0 = x - velocity * time;
		while (x0 < prob_lo_x) {
			x0 += length;
		}
		while (x0 >= prob_hi_x) {
			x0 -= length;
		}
		state(i, j, k, 0) = sawtooth(x0, prob_lo_x, prob_hi_x);
	});
}

void initializeHook(quokka::grid const &grid_elem)
{
	exactSolution(grid_elem, 0.0);
}

void diagnosticsHook(AdvectionSimulationPrototype &sim)
{
	if (!amrex::ParallelDescriptor::IOProcessor()) {
		return;
	}
	amrex::Print() << "Prototype time = " << sim.currentTime() << ", L1(error) = " << sim.errorNorm() << "\n";
}

} // namespace

auto problem_main() -> int
{
	constexpr amrex::Real prob_lo = 0.0;
	constexpr amrex::Real prob_hi = 1.0;
	constexpr int nx = 6400;
	constexpr amrex::Real advection_velocity = 1.0;
	constexpr amrex::Real max_time = 1.0;
	constexpr amrex::Real cfl = 0.4;
	constexpr amrex::Real err_tol = 9.0e-3;

	g_problem.prob_lo = prob_lo;
	g_problem.prob_hi = prob_hi;
	g_problem.advection_velocity = advection_velocity;

	AdvectionSimulationPrototype sim;

	sim.configureGrid(nx, prob_lo, prob_hi);
	sim.setAdvectionVelocity(advection_velocity, 0.0, 0.0);
	sim.setCfl(cfl);
	sim.setMaxTime(max_time);
	sim.setMaxTimesteps(5000);

	AdvectionSimulationPrototype::Callbacks callbacks;
	callbacks.hooks.exact_solution = exactSolution;
	callbacks.hooks.initialize = initializeHook;
	callbacks.hooks.diagnostics = diagnosticsHook;
	sim.setCallbacks(std::move(callbacks));

	sim.run();
	const amrex::Real err = sim.errorNorm();
	if (amrex::ParallelDescriptor::IOProcessor()) {
		amrex::Print() << "Prototype advection error norm = " << err << "\n";
	}
	return (err < err_tol) ? 0 : 1;
}
