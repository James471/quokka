#include "experimental/DescriptorPrototype.hpp"
#include "AMReX.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include <cmath>

using quokka::experimental::AdvectionSimulationPrototype;
using quokka::experimental::HydroModule;
using quokka::experimental::PhysicsDescriptor;
using quokka::experimental::PhysicsLayout;
using quokka::experimental::ProblemHooks;
using quokka::experimental::setRuntimePhysicsLayout;

namespace
{

AMREX_FORCE_INLINE auto sawtooth(amrex::Real x, amrex::Real prob_lo, amrex::Real prob_hi) -> amrex::Real
{
	const amrex::Real length = prob_hi - prob_lo;
	amrex::Real shifted = x - prob_lo;
	shifted = std::fmod(shifted + length, length);
	return std::fmod(shifted + 0.5 * length, length);
}

} // namespace

auto main(int argc, char **argv) -> int
{
	amrex::Initialize(argc, argv);
	int status = 0;
	{
		constexpr amrex::Real prob_lo = 0.0;
		constexpr amrex::Real prob_hi = 1.0;
		constexpr int nx = 6400;
		constexpr amrex::Real advection_velocity = 1.0;
		constexpr amrex::Real max_time = 1.0;
		constexpr amrex::Real cfl = 0.4;
		constexpr amrex::Real err_tol = 9.0e-3;

		PhysicsLayout layout;
		layout.total_cc_components = 1;
		layout.rad_group_stride = 0;
		layout.scalar_stride = 1;
		setRuntimePhysicsLayout(layout);

		AdvectionSimulationPrototype sim(layout);
		sim.configureGrid(nx, prob_lo, prob_hi);
		sim.setAdvectionVelocity(advection_velocity, 0.0, 0.0);
		sim.setCfl(cfl);
		sim.setMaxTime(max_time);
		sim.setMaxTimesteps(5000);

		sim.setInitialCondition([](amrex::Real x) { return sawtooth(x, prob_lo, prob_hi); });

		sim.setExactSolution([=](amrex::Real x, amrex::Real time) {
			const amrex::Real length = prob_hi - prob_lo;
			amrex::Real x0 = x - advection_velocity * time;
			while (x0 < prob_lo) {
				x0 += length;
			}
			while (x0 >= prob_hi) {
				x0 -= length;
			}
			return sawtooth(x0, prob_lo, prob_hi);
		});

		ProblemHooks hooks;
		hooks.diagnostics = [&sim]() {
			if (amrex::ParallelDescriptor::IOProcessor()) {
				amrex::Print() << "Prototype time = " << sim.currentTime() << ", L1(error) = " << sim.errorNorm() << "\n";
			}
		};
		sim.setHooks(hooks);

		sim.run();
		const amrex::Real err = sim.errorNorm();
		if (amrex::ParallelDescriptor::IOProcessor()) {
			amrex::Print() << "Prototype advection error norm = " << err << "\n";
		}
		status = (err < err_tol) ? 0 : 1;
	}
	amrex::Finalize();
	return status;
}
