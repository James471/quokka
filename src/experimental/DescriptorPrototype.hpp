#ifndef QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_
#define QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_

#include "AMReX_Array.H"
#include "AMReX_Array4.H"
#include "AMReX_AmrCore.H"
#include "AMReX_Box.H"
#include "AMReX_Dim3.H"
#include "AMReX_GpuQualifiers.H"
#include "AMReX_MultiFab.H"
#include "AMReX_TagBox.H"
#include "AMReX_Vector.H"
#include "grid.hpp"
#include "physics_numVars.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace quokka::experimental
{

struct HydroModule {
};
struct RadiationModule {
};
struct MHDModule {
};

struct PhysicsLayout {
	int num_rad_groups = 0;
	int rad_offset = 0;
	int rad_group_stride = 0;

	int num_mass_scalars = 0;
	int mass_scalar_offset = 0;

	int num_passive_scalars = 0;
	int passive_scalar_offset = 0;

	int total_cc_components = 0;
	int scalar_stride = 1;
};

inline PhysicsLayout runtime_physics_layout{};

inline void setRuntimePhysicsLayout(const PhysicsLayout &layout) { runtime_physics_layout = layout; }
inline auto getRuntimePhysicsLayout() -> PhysicsLayout const & { return runtime_physics_layout; }

template <typename... Modules> struct PhysicsDescriptor {
	static constexpr bool has_hydro = (std::is_same_v<Modules, HydroModule> || ...);
	static constexpr bool has_radiation = (std::is_same_v<Modules, RadiationModule> || ...);
	static constexpr bool has_mhd = (std::is_same_v<Modules, MHDModule> || ...);

	static constexpr int hydro_cc_size = has_hydro ? Physics_NumVars::numHydroVars : 0;
	static constexpr int rad_group_stride = has_radiation ? Physics_NumVars::numRadVarsPerGroup : 0;
	static constexpr int scalar_stride = 1;
	static constexpr int face_mhd_stride = has_mhd ? Physics_NumVars::numMHDVars_per_dim : 0;

	AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static auto layout() -> PhysicsLayout const & { return runtime_physics_layout; }
};

class AMRSimulationPrototype;

struct ProblemHooks {
	using InitializeFunc = std::function<void(quokka::grid const &)>;
	using ExactSolutionFunc = std::function<void(quokka::grid const &, amrex::Real)>;
	using HookFunc = std::function<void(AMRSimulationPrototype &)>;

	InitializeFunc initialize;
	ExactSolutionFunc exact_solution;
	HookFunc pre_timestep;
	HookFunc post_timestep;
	HookFunc diagnostics;
};

template <typename SimulationT> struct TypedProblemHooks {
	using InitializeFunc = ProblemHooks::InitializeFunc;
	using ExactSolutionFunc = ProblemHooks::ExactSolutionFunc;
	using HookFunc = std::function<void(SimulationT &)>;

	InitializeFunc initialize;
	ExactSolutionFunc exact_solution;
	HookFunc pre_timestep;
	HookFunc post_timestep;
	HookFunc diagnostics;
};

class AMRSimulationPrototype : public amrex::AmrCore
{
      public:
	explicit AMRSimulationPrototype(PhysicsLayout layout);
	virtual ~AMRSimulationPrototype() = default;

	void setHooks(ProblemHooks hooks);
	template <typename SimulationT> void setHooks(TypedProblemHooks<SimulationT> hooks)
	{
		ProblemHooks base_hooks;
		if (hooks.initialize) {
			base_hooks.initialize = std::move(hooks.initialize);
		}
		if (hooks.exact_solution) {
			base_hooks.exact_solution = std::move(hooks.exact_solution);
		}
		if (hooks.pre_timestep) {
			base_hooks.pre_timestep = [func = std::move(hooks.pre_timestep)](AMRSimulationPrototype &sim) {
				func(static_cast<SimulationT &>(sim));
			};
		}
		if (hooks.post_timestep) {
			base_hooks.post_timestep = [func = std::move(hooks.post_timestep)](AMRSimulationPrototype &sim) {
				func(static_cast<SimulationT &>(sim));
			};
		}
		if (hooks.diagnostics) {
			base_hooks.diagnostics = [func = std::move(hooks.diagnostics)](AMRSimulationPrototype &sim) { func(static_cast<SimulationT &>(sim)); };
		}
		setHooks(std::move(base_hooks));
	}

	[[nodiscard]] auto layout() const -> PhysicsLayout const & { return layout_; }

	virtual void step() = 0;

      protected:
	void setBaseGrid(int nx, amrex::Real prob_lo, amrex::Real prob_hi);
	void applyInitializeHook(int lev, amrex::MultiFab &mf) const;
	void applyExactSolutionHook(int lev, amrex::MultiFab &mf, amrex::Real time) const;
	[[nodiscard]] auto cellData(int lev) -> amrex::MultiFab &;
	[[nodiscard]] auto cellData(int lev) const -> amrex::MultiFab const &;
	[[nodiscard]] auto scratchData(int lev) -> amrex::MultiFab &;
	[[nodiscard]] auto scratchData(int lev) const -> amrex::MultiFab const &;
	[[nodiscard]] auto gridIsConfigured() const -> bool { return grid_configured_; }

	void ErrorEst(int lev, amrex::TagBoxArray &tags, amrex::Real time, int ngrow) override;
	void MakeNewLevelFromScratch(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;
	void MakeNewLevelFromCoarse(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;
	void RemakeLevel(int lev, amrex::Real time, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm) override;
	void ClearLevel(int lev) override;

	void allocateLevelData(int lev, const amrex::BoxArray &ba, const amrex::DistributionMapping &dm);

	PhysicsLayout layout_;
	ProblemHooks hooks_;
	amrex::Array<int, AMREX_SPACEDIM> periodicity_{AMREX_D_DECL(1, 0, 0)};
	amrex::Vector<std::unique_ptr<amrex::MultiFab>> state_cc_;
	amrex::Vector<std::unique_ptr<amrex::MultiFab>> scratch_cc_;
	int nghost_cc_ = 1;
	bool grid_configured_ = false;
};

class AdvectionSimulationPrototype : public AMRSimulationPrototype
{
      public:
	using Descriptor = PhysicsDescriptor<HydroModule>;

	static auto defaultLayout() -> PhysicsLayout;

	AdvectionSimulationPrototype();
	explicit AdvectionSimulationPrototype(PhysicsLayout layout);

	void setAdvectionVelocity(amrex::Real vx, amrex::Real vy, amrex::Real vz);
	void configureGrid(int nx, amrex::Real prob_lo, amrex::Real prob_hi);
	void setCfl(amrex::Real cfl);
	void setMaxTime(amrex::Real stop_time);
	void setMaxTimesteps(int steps);

	struct Callbacks {
		TypedProblemHooks<AdvectionSimulationPrototype> hooks;
	};

	void setCallbacks(Callbacks callbacks);

	void initializeState();

	void step() override;
	void run();

	[[nodiscard]] auto estimateMaxSignalSpeed() const -> amrex::Real;
	[[nodiscard]] auto errorNorm() const -> amrex::Real { return error_norm_; }
	[[nodiscard]] auto currentTime() const -> amrex::Real { return time_; }
	[[nodiscard]] auto probLo() const -> amrex::Real { return prob_lo_; }
	[[nodiscard]] auto probHi() const -> amrex::Real { return prob_hi_; }
	[[nodiscard]] auto dx() const -> amrex::Real { return dx_; }

      private:
	void advance(amrex::Real dt);
	void computeError();

	amrex::GpuArray<amrex::Real, 3> velocity_{0., 0., 0.};
	int nx_ = 0;
	amrex::Real prob_lo_ = 0.;
	amrex::Real prob_hi_ = 1.;
	amrex::Real dx_ = 1.;
	amrex::Real cfl_ = 0.4;
	amrex::Real stop_time_ = 0.;
	int max_steps_ = 1000;

	amrex::Real time_ = 0.;

	bool state_initialized_ = false;

	amrex::Real error_norm_ = 0.;
};

} // namespace quokka::experimental

#endif // QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_
