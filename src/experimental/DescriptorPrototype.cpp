#include "experimental/DescriptorPrototype.hpp"
#include "AMReX_DistributionMapping.H"
#include "AMReX_Math.H"
#include "AMReX_ParallelDescriptor.H"
#include "AMReX_Print.H"
#include "AMReX_Reduce.H"
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
	state_cc_.clear();
	scratch_cc_.clear();
	grid_configured_ = false;
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
	grid_configured_ = true;
}

void AMRSimulationPrototype::allocateLevelData(int lev, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	const int num_levels = lev + 1;
	if (static_cast<int>(state_cc_.size()) < num_levels) {
		state_cc_.resize(num_levels);
		scratch_cc_.resize(num_levels);
	}
	const int ncomp = layout_.total_cc_components;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(ncomp > 0, "AMRSimulationPrototype requires at least one cell-centered component.");
	state_cc_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost_cc_);
	scratch_cc_[lev] = std::make_unique<amrex::MultiFab>(ba, dm, ncomp, nghost_cc_);
	state_cc_[lev]->setVal(0.0);
	scratch_cc_[lev]->setVal(0.0);
}

auto AMRSimulationPrototype::cellData(int lev) -> amrex::MultiFab &
{
	if ((lev >= static_cast<int>(state_cc_.size())) || (state_cc_[lev] == nullptr)) {
		throw std::runtime_error("cellData: level data not allocated.");
	}
	return *state_cc_[lev];
}

auto AMRSimulationPrototype::cellData(int lev) const -> amrex::MultiFab const &
{
	if ((lev >= static_cast<int>(state_cc_.size())) || (state_cc_[lev] == nullptr)) {
		throw std::runtime_error("cellData: level data not allocated.");
	}
	return *state_cc_[lev];
}

auto AMRSimulationPrototype::scratchData(int lev) -> amrex::MultiFab &
{
	if ((lev >= static_cast<int>(scratch_cc_.size())) || (scratch_cc_[lev] == nullptr)) {
		throw std::runtime_error("scratchData: level data not allocated.");
	}
	return *scratch_cc_[lev];
}

auto AMRSimulationPrototype::scratchData(int lev) const -> amrex::MultiFab const &
{
	if ((lev >= static_cast<int>(scratch_cc_.size())) || (scratch_cc_[lev] == nullptr)) {
		throw std::runtime_error("scratchData: level data not allocated.");
	}
	return *scratch_cc_[lev];
}

void AMRSimulationPrototype::applyInitializeHook(int lev, amrex::MultiFab &mf) const
{
	if (!hooks_.initialize) {
		mf.setVal(0.0);
		return;
	}
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		quokka::grid grid_elem(mf.array(iter), iter.validbox(), Geom(lev).CellSizeArray(), Geom(lev).ProbLoArray(), Geom(lev).ProbHiArray(),
				       quokka::centering::cc, quokka::direction::na);
		hooks_.initialize(grid_elem);
	}
	mf.FillBoundary(Geom(lev).periodicity());
}

void AMRSimulationPrototype::applyExactSolutionHook(int lev, amrex::MultiFab &mf, amrex::Real time) const
{
	if (!hooks_.exact_solution) {
		mf.setVal(0.0);
		return;
	}
	for (amrex::MFIter iter(mf); iter.isValid(); ++iter) {
		quokka::grid grid_elem(mf.array(iter), iter.validbox(), Geom(lev).CellSizeArray(), Geom(lev).ProbLoArray(), Geom(lev).ProbHiArray(),
				       quokka::centering::cc, quokka::direction::na);
		hooks_.exact_solution(grid_elem, time);
	}
	mf.FillBoundary(Geom(lev).periodicity());
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
	(void)time;
	allocateLevelData(lev, ba, dm);
}

void AMRSimulationPrototype::MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	(void)time;
	AMREX_ALWAYS_ASSERT_WITH_MESSAGE(lev == 0, "MakeNewLevelFromCoarse not implemented for lev > 0 in prototype.");
	allocateLevelData(lev, ba, dm);
}

void AMRSimulationPrototype::RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm)
{
	(void)time;
	allocateLevelData(lev, ba, dm);
}

void AMRSimulationPrototype::ClearLevel(int lev)
{
	if (lev < static_cast<int>(state_cc_.size())) {
		state_cc_[lev].reset();
	}
	if (lev < static_cast<int>(scratch_cc_.size())) {
		scratch_cc_[lev].reset();
	}
}

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
	if (!gridIsConfigured()) {
		throw std::runtime_error("initializeState: grid not configured.");
	}
	const int lev = 0;
	MakeNewLevelFromScratch(lev, 0.0, boxArray(lev), DistributionMap(lev));
	auto &state = cellData(lev);
	applyInitializeHook(lev, state);
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
	if (!gridIsConfigured()) {
		throw std::runtime_error("advance: grid not configured.");
	}
	auto &state = cellData(0);
	auto &scratch = scratchData(0);
	const amrex::Real vel = velocity_[0];
	const amrex::Real abs_v = std::abs(vel);
	if (abs_v == 0.0) {
		return;
	}

	const amrex::Real lambda = abs_v * dt / dx_;
	state.FillBoundary(Geom(0).periodicity());

	for (amrex::MFIter iter(state); iter.isValid(); ++iter) {
		const amrex::Box &box = iter.validbox();
		const auto state_arr = state.array(iter);
		const auto scratch_arr = scratch.array(iter);
		amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
			if (vel >= 0.0) {
				scratch_arr(i, j, k, 0) = state_arr(i, j, k, 0) - lambda * (state_arr(i, j, k, 0) - state_arr(i - 1, j, k, 0));
			} else {
				scratch_arr(i, j, k, 0) = state_arr(i, j, k, 0) - lambda * (state_arr(i + 1, j, k, 0) - state_arr(i, j, k, 0));
			}
		});
	}
	amrex::MultiFab::Copy(state, scratch, 0, 0, state.nComp(), state.nGrow());
}

void AdvectionSimulationPrototype::computeError()
{
	if (!hooks_.exact_solution) {
		error_norm_ = 0.0;
		return;
	}
	auto &state = cellData(0);
	auto &scratch = scratchData(0);
	applyExactSolutionHook(0, scratch, time_);
	amrex::MultiFab diff(state.boxArray(), state.DistributionMap(), state.nComp(), 0);
	amrex::MultiFab::Copy(diff, state, 0, 0, state.nComp(), 0);
	diff.minus(scratch, 0, state.nComp(), 0);
	amrex::Real l1 = 0.0;
	for (int comp = 0; comp < state.nComp(); ++comp) {
		l1 += diff.norm1(comp, 0);
	}
	const amrex::Real num_cells = static_cast<amrex::Real>(boxArray(0).numPts());
	error_norm_ = (num_cells > 0.0) ? (l1 / num_cells) : 0.0;
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
