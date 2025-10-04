#ifndef DIAGBASE_H
#define DIAGBASE_H

#include <unordered_map>

#include "AMReX_MultiFab.H"
#include "AMReX_Orientation.H"
#include "DiagFilter.H"
#include "Factory.H"
#include "yaml-cpp/yaml.h"

// Forward declarations
template <typename problem_t> class AMRSimulation;
class DiagPlotfile;
class DiagProjectionPlot;
class DiagFramePlane;
class DiagPDF;

class DiagBase : public quokka::Factory<DiagBase>
{
      public:
	~DiagBase() override = default;

	static auto base_identifier() -> std::string { return "DiagBase"; }

	virtual void init(const std::string &a_prefix, std::string_view a_diagName);

	virtual void close() = 0;

	[[nodiscard]] auto needUpdate() const -> bool { return need_update; }

	virtual auto doDiag(const amrex::Real &a_time, int a_nstep) -> bool;

	virtual void prepare(int a_nlevels, const amrex::Vector<amrex::Geometry> &a_geoms, const amrex::Vector<amrex::BoxArray> &a_grids,
			     const amrex::Vector<amrex::DistributionMapping> &a_dmap, const amrex::Vector<std::string> &a_varNames);

	// Type for projection data (used by DiagProjectionPlot)
	using ProjectionData = std::unordered_map<std::string, amrex::BaseFab<amrex::Real>>;

	// Pure virtual processDiag - each diagnostic implements this to process diagnostic output
	virtual void processDiag(int a_nstep, const amrex::Real &a_time) = 0;

	virtual void addVars(amrex::Vector<std::string> &a_varList);

	static auto getFieldIndex(const std::string &a_field, const amrex::Vector<std::string> &a_varList) -> int;

	static auto getFieldIndexVec(const std::vector<std::string> &a_field, const amrex::Vector<std::string> &a_varList) -> amrex::Vector<int>;

	// Set diagnostic data - called by AMRSimulation::doDiagnostics()
	// This must be implemented in simulation.hpp after all diagnostic headers are included
	template <typename problem_t>
	void setDiagData(AMRSimulation<problem_t> *sim, const amrex::Vector<const amrex::MultiFab *> *diagMF, const amrex::Vector<std::string> *diagVars,
			 const amrex::Vector<amrex::Geometry> *geoms, const amrex::Vector<amrex::IntVect> *refRatio, const YAML::Node *metadata);

      protected:
	std::string m_diagfile;
	int m_verbose{0};
	amrex::Real m_per{-1.0};
	int m_interval{-1};
	bool need_update{true};
	bool first_time{true};
	amrex::Vector<DiagFilter> m_filters{};
	amrex::Gpu::DeviceVector<DiagFilterData> m_filterData;

	// Pointer to the AMRSimulation instance (set by setDiagData)
	void *m_simulation{nullptr};

	// Pointers to commonly used diagnostic data (set by setDiagData)
	const amrex::Vector<const amrex::MultiFab *> *m_diagMF{nullptr};
	const amrex::Vector<std::string> *m_diagVars{nullptr};
	const amrex::Vector<amrex::Geometry> *m_geoms{nullptr};
	const amrex::Vector<amrex::IntVect> *m_refRatio{nullptr};
	const YAML::Node *m_metadata{nullptr};

	// Helper to get simulation pointer with correct type
	template <typename problem_t> auto getSim() const -> AMRSimulation<problem_t> * { return static_cast<AMRSimulation<problem_t> *>(m_simulation); }

	// Make AMRSimulation a friend so derived diagnostic classes can access its members
	template <typename problem_t> friend class AMRSimulation;
};

#endif
