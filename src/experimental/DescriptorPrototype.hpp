#ifndef QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_
#define QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_

#include "AMReX_Array.H"
#include "AMReX_GpuQualifiers.H"
#include "physics_numVars.hpp"
#include <functional>
#include <stdexcept>
#include <tuple>
#include <type_traits>
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

struct ProblemHooks {
	std::function<void()> pre_timestep;
	std::function<void()> post_timestep;
	std::function<void()> diagnostics;
};

class AMRSimulationPrototype {
      public:
	explicit AMRSimulationPrototype(PhysicsLayout layout);
	virtual ~AMRSimulationPrototype() = default;

	void setHooks(ProblemHooks hooks);

	[[nodiscard]] auto layout() const -> PhysicsLayout const & { return layout_; }

	virtual void step() = 0;

      protected:
	PhysicsLayout layout_;
	ProblemHooks hooks_;
};

class AdvectionSimulationPrototype : public AMRSimulationPrototype {
      public:
	using Descriptor = PhysicsDescriptor<HydroModule>;

	explicit AdvectionSimulationPrototype(PhysicsLayout layout);

	void setAdvectionVelocity(amrex::Real vx, amrex::Real vy, amrex::Real vz);
	void configureGrid(int nx, amrex::Real prob_lo, amrex::Real prob_hi);
	void setCfl(amrex::Real cfl);
	void setMaxTime(amrex::Real stop_time);
	void setMaxTimesteps(int steps);

	using InitialConditionFunc = std::function<amrex::Real(amrex::Real)>;
	using ExactSolutionFunc = std::function<amrex::Real(amrex::Real, amrex::Real)>;

	void setInitialCondition(InitialConditionFunc func);
	void setExactSolution(ExactSolutionFunc func);

	void initializeState();

	void step() override;
	void run();

	[[nodiscard]] auto estimateMaxSignalSpeed() const -> amrex::Real;
	[[nodiscard]] auto errorNorm() const -> amrex::Real { return error_norm_; }
	[[nodiscard]] auto currentTime() const -> amrex::Real { return time_; }
	[[nodiscard]] auto state() const -> std::vector<amrex::Real> const & { return state_; }

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

	InitialConditionFunc init_func_;
	ExactSolutionFunc exact_func_;
	bool state_initialized_ = false;
	bool exact_available_ = false;

	std::vector<amrex::Real> state_;
	std::vector<amrex::Real> scratch_;

	amrex::Real error_norm_ = 0.;
};

} // namespace quokka::experimental

#endif // QUOKKA_DESCRIPTOR_PROTOTYPE_HPP_
