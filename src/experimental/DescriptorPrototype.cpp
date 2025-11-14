#include "experimental/DescriptorPrototype.hpp"
#include "AMReX_Print.H"
#include <cmath>
#include <utility>

namespace quokka::experimental
{

AMRSimulationPrototype::AMRSimulationPrototype(PhysicsLayout layout) : layout_(layout) { setRuntimePhysicsLayout(layout_); }

void AMRSimulationPrototype::setHooks(ProblemHooks hooks) { hooks_ = std::move(hooks); }

AdvectionSimulationPrototype::AdvectionSimulationPrototype(PhysicsLayout layout) : AMRSimulationPrototype(layout) {}

void AdvectionSimulationPrototype::setAdvectionVelocity(amrex::Real vx, amrex::Real vy, amrex::Real vz)
{
	velocity_[0] = vx;
	velocity_[1] = vy;
	velocity_[2] = vz;
}

void AdvectionSimulationPrototype::configureGrid(int nx, amrex::Real prob_lo, amrex::Real prob_hi)
{
	if (nx <= 0) {
		throw std::runtime_error("configureGrid: nx must be positive.");
	}
	nx_ = nx;
	prob_lo_ = prob_lo;
	prob_hi_ = prob_hi;
	dx_ = (prob_hi_ - prob_lo_) / static_cast<amrex::Real>(nx_);
	state_.assign(nx_, 0.0);
	scratch_ = state_;
	state_initialized_ = false;
}

void AdvectionSimulationPrototype::setCfl(amrex::Real cfl)
{
	if ((cfl <= 0.0) || (cfl > 1.0)) {
		throw std::runtime_error("setCfl: CFL must be in (0, 1].");
	}
	cfl_ = cfl;
}

void AdvectionSimulationPrototype::setMaxTime(amrex::Real stop_time)
{
	if (stop_time <= 0.0) {
		throw std::runtime_error("setMaxTime: stop_time must be positive.");
	}
	stop_time_ = stop_time;
}

void AdvectionSimulationPrototype::setMaxTimesteps(int steps)
{
	if (steps <= 0) {
		throw std::runtime_error("setMaxTimesteps: steps must be positive.");
	}
	max_steps_ = steps;
}

void AdvectionSimulationPrototype::setInitialCondition(InitialConditionFunc func)
{
	init_func_ = std::move(func);
	state_initialized_ = false;
}

void AdvectionSimulationPrototype::setExactSolution(ExactSolutionFunc func)
{
	exact_func_ = std::move(func);
	exact_available_ = static_cast<bool>(exact_func_);
}

void AdvectionSimulationPrototype::initializeState()
{
	if (!init_func_) {
		throw std::runtime_error("initializeState: initial condition callback not set.");
	}
	if (nx_ <= 0) {
		throw std::runtime_error("initializeState: grid not configured.");
	}
	for (int i = 0; i < nx_; ++i) {
		amrex::Real const x = prob_lo_ + (static_cast<amrex::Real>(i) + 0.5) * dx_;
		state_[i] = init_func_(x);
	}
	time_ = 0.0;
	state_initialized_ = true;
	error_norm_ = 0.0;
}

auto AdvectionSimulationPrototype::estimateMaxSignalSpeed() const -> amrex::Real
{
	const amrex::Real speed = std::sqrt(velocity_[0] * velocity_[0] + velocity_[1] * velocity_[1] + velocity_[2] * velocity_[2]);
	return speed;
}

void AdvectionSimulationPrototype::advance(amrex::Real dt)
{
	if (nx_ == 0) {
		throw std::runtime_error("advance: grid not configured.");
	}

	const amrex::Real vel = velocity_[0];
	const amrex::Real abs_v = std::abs(vel);
	if (abs_v == 0.0) {
		return;
	}

	const amrex::Real lambda = abs_v * dt / dx_;

	for (int i = 0; i < nx_; ++i) {
		int left = (i - 1 + nx_) % nx_;
		int right = (i + 1) % nx_;
		if (vel >= 0.0) {
			scratch_[i] = state_[i] - lambda * (state_[i] - state_[left]);
		} else {
			scratch_[i] = state_[i] - lambda * (state_[right] - state_[i]);
		}
	}

	state_.swap(scratch_);
}

void AdvectionSimulationPrototype::computeError()
{
	if (!exact_available_) {
		error_norm_ = 0.0;
		return;
	}
	amrex::Real l1 = 0.0;
	for (int i = 0; i < nx_; ++i) {
		amrex::Real const x = prob_lo_ + (static_cast<amrex::Real>(i) + 0.5) * dx_;
		amrex::Real const exact = exact_func_(x, time_);
		l1 += std::abs(state_[i] - exact);
	}
	error_norm_ = l1 / static_cast<amrex::Real>(nx_);
}

void AdvectionSimulationPrototype::step()
{
	if (!state_initialized_) {
		initializeState();
	}

	const amrex::Real abs_v = std::abs(velocity_[0]);
	if (abs_v == 0.0) {
		return;
	}
	amrex::Real dt = cfl_ * dx_ / abs_v;
	if (dt <= 0.0) {
		throw std::runtime_error("step: invalid timestep (<= 0).");
	}
	if (hooks_.pre_timestep) {
		hooks_.pre_timestep();
	}
	if (time_ + dt > stop_time_) {
		dt = stop_time_ - time_;
	}
	advance(dt);
	time_ += dt;
	if (hooks_.post_timestep) {
		hooks_.post_timestep();
	}
}

void AdvectionSimulationPrototype::run()
{
	if (!state_initialized_) {
		initializeState();
	}
	if (stop_time_ <= 0.0) {
		throw std::runtime_error("run: stop_time not configured.");
	}

	int nstep = 0;
	while ((time_ < stop_time_) && (nstep < max_steps_)) {
		step();
		++nstep;
	}

	computeError();
	if (hooks_.diagnostics) {
		hooks_.diagnostics();
	}
}

} // namespace quokka::experimental
