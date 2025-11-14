#include "experimental/DescriptorPrototype.hpp"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Print.H"
#include <cmath>
#include <algorithm>
#include <array>
#include <utility>

namespace quokka::experimental
{

namespace
{

auto makeDefaultGeometry() -> amrex::Geometry
{
	amrex::IntVect lo = amrex::IntVect::TheZeroVector();
	amrex::IntVect hi = amrex::IntVect::TheZeroVector();
	amrex::Box domain(lo, hi);
	std::array<amrex::Real, AMREX_SPACEDIM> rb_lo{AMREX_D_DECL(0.0, 0.0, 0.0)};
	std::array<amrex::Real, AMREX_SPACEDIM> rb_hi{AMREX_D_DECL(1.0, 1.0, 1.0)};
	amrex::RealBox real_box(rb_lo, rb_hi);
	amrex::Array<int, AMREX_SPACEDIM> periodic{AMREX_D_DECL(1, 0, 0)};
	return amrex::Geometry(domain, &real_box, 0, periodic.data());
}

auto makeDefaultAmrInfo() -> amrex::AmrInfo
{
	amrex::AmrInfo info;
	info.max_level = 0;
	info.ref_ratio = {amrex::IntVect::TheUnitVector()};
	info.blocking_factor = {amrex::IntVect::TheUnitVector()};
	info.max_grid_size = {amrex::IntVect::TheUnitVector()};
	info.n_error_buf = {amrex::IntVect::TheUnitVector()};
	info.check_input = false;
	return info;
}

} // namespace

AMRSimulationPrototype::AMRSimulationPrototype(PhysicsLayout layout)
    : amrex::AmrCore(makeDefaultGeometry(), makeDefaultAmrInfo()), layout_(layout)
{
	setRuntimePhysicsLayout(layout_);
	setBaseGrid(1, 0.0, 1.0);
}

void AMRSimulationPrototype::setHooks(ProblemHooks hooks) { hooks_ = std::move(hooks); }

void AMRSimulationPrototype::setBaseGrid(int nx, amrex::Real prob_lo, amrex::Real prob_hi)
{
	const int cells_x = std::max(nx, 1);
	amrex::IntVect lo = amrex::IntVect::TheZeroVector();
	amrex::IntVect hi = amrex::IntVect::TheZeroVector();
	hi[0] = cells_x - 1;
#if (AMREX_SPACEDIM >= 2)
	hi[1] = 0;
#endif
#if (AMREX_SPACEDIM == 3)
	hi[2] = 0;
#endif
	amrex::Box domain(lo, hi);
	std::array<amrex::Real, AMREX_SPACEDIM> rb_lo{AMREX_D_DECL(prob_lo, 0.0, 0.0)};
	std::array<amrex::Real, AMREX_SPACEDIM> rb_hi{AMREX_D_DECL(prob_hi, 1.0, 1.0)};
	amrex::RealBox real_box(rb_lo, rb_hi);
	amrex::Geometry geom(domain, &real_box, 0, periodicity_.data());
	SetGeometry(0, geom);
	amrex::BoxArray ba(domain);
	amrex::DistributionMapping dm(ba);
	SetBoxArray(0, ba);
	SetDistributionMap(0, dm);
	SetFinestLevel(0);
}

void AMRSimulationPrototype::ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow)
{
	(void)lev;
	(void)tags;
	(void)time;
	(void)ngrow;
}

void AMRSimulationPrototype::MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	(void)lev;
	(void)time;
	(void)ba;
	(void)dm;
}

void AMRSimulationPrototype::MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	(void)lev;
	(void)time;
	(void)ba;
	(void)dm;
}

void AMRSimulationPrototype::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	(void)lev;
	(void)time;
	(void)ba;
	(void)dm;
}

void AMRSimulationPrototype::ClearLevel(int lev) { (void)lev; }

auto AdvectionSimulationPrototype::defaultLayout() -> PhysicsLayout
{
	PhysicsLayout layout{};
	layout.total_cc_components = 1;
	layout.rad_group_stride = 0;
	layout.scalar_stride = 1;
	return layout;
}

AdvectionSimulationPrototype::AdvectionSimulationPrototype() : AdvectionSimulationPrototype(defaultLayout()) {}

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
	setBaseGrid(nx_, prob_lo_, prob_hi_);
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

void AdvectionSimulationPrototype::setCallbacks(Callbacks callbacks)
{
	const bool has_init_hook = static_cast<bool>(callbacks.hooks.initialize);
	AMRSimulationPrototype::setHooks(std::move(callbacks.hooks));
	if (has_init_hook) {
		state_initialized_ = false;
	}
}

void AdvectionSimulationPrototype::initializeState()
{
	if (!hooks_.initialize) {
		throw std::runtime_error("initializeState: initial condition callback not set.");
	}
	if (nx_ <= 0) {
		throw std::runtime_error("initializeState: grid not configured.");
	}
	if (static_cast<int>(state_.size()) != nx_) {
		state_.assign(nx_, 0.0);
	}
	scratch_.resize(nx_);
	auto grid_elem = buildGridView(state_);
	hooks_.initialize(grid_elem);
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
	if (!hooks_.exact_solution) {
		error_norm_ = 0.0;
		return;
	}
	auto grid_exact = buildGridView(scratch_);
	hooks_.exact_solution(grid_exact, time_);
	amrex::Real l1 = 0.0;
	for (int i = 0; i < nx_; ++i) {
		l1 += std::abs(state_[i] - scratch_[i]);
	}
	error_norm_ = l1 / static_cast<amrex::Real>(nx_);
}

auto AdvectionSimulationPrototype::buildGridView(amrex::Vector<amrex::Real> &storage) -> quokka::grid
{
	if (storage.empty()) {
		throw std::runtime_error("buildGridView: state storage not allocated.");
	}
	amrex::IntVect small = amrex::IntVect::TheZeroVector();
	amrex::IntVect big = amrex::IntVect::TheZeroVector();
	big[0] = (nx_ > 0) ? (nx_ - 1) : 0;
	amrex::Box box(small, big);
	amrex::Dim3 begin{small[0], 0, 0};
	amrex::Dim3 end{big[0] + 1, 1, 1};
#if (AMREX_SPACEDIM >= 2)
	begin.y = small[1];
	end.y = big[1] + 1;
#endif
#if (AMREX_SPACEDIM == 3)
	begin.z = small[2];
	end.z = big[2] + 1;
#endif
	auto *raw = reinterpret_cast<double *>(storage.data());
	amrex::Array4<double> array(raw, begin, end, 1);

	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> dx{};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_lo{};
	amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> prob_hi{};
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		dx[d] = (d == 0) ? dx_ : 1.0;
		prob_lo[d] = (d == 0) ? prob_lo_ : 0.0;
		prob_hi[d] = (d == 0) ? prob_hi_ : 1.0;
	}

	return quokka::grid(array, box, dx, prob_lo, prob_hi, quokka::centering::cc, quokka::direction::x);
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
		hooks_.pre_timestep(*this);
	}
	if (time_ + dt > stop_time_) {
		dt = stop_time_ - time_;
	}
	advance(dt);
	time_ += dt;
	if (hooks_.post_timestep) {
		hooks_.post_timestep(*this);
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
		hooks_.diagnostics(*this);
	}
}

} // namespace quokka::experimental
