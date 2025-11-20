#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <utility>

#include "AMReX.H"
#include "AMReX_Geometry.H"
#include "Base/Array4.H"
#include "experimental/DescriptorPrototype.hpp"
#include "grid.hpp"
#include "pyAMReX.H"

namespace py = pybind11;
using quokka::experimental::AdvectionSimulationPrototype;

namespace
{

constexpr auto amrexPythonModuleName()
{
#if AMREX_SPACEDIM == 1
	return "amrex.space1d";
#elif AMREX_SPACEDIM == 2
	return "amrex.space2d";
#else
	return "amrex.space3d";
#endif
}

void ensureArray4Bindings(py::module_ &m)
{
	auto *type_info = py::detail::get_type_info(typeid(amrex::Array4<amrex::Real>));
	if (type_info != nullptr) {
		return;
	}

	try {
		py::module_::import(amrexPythonModuleName());
	} catch (py::error_already_set &) {
		PyErr_Clear();
	}

	type_info = py::detail::get_type_info(typeid(amrex::Array4<amrex::Real>));
	if (type_info != nullptr) {
		return;
	}

	pyAMReX::make_Array4<amrex::Real>(m, "double");
	pyAMReX::make_Array4<amrex::Real const>(m, "double_const");
}

auto makeGridObject(quokka::grid const &grid) -> py::object
{
	return py::cast(&grid, py::return_value_policy::reference);
}

auto makeRangePair(const amrex::Box &box, int dir) -> std::pair<int, int>
{
	return std::make_pair(box.smallEnd(dir), box.bigEnd(dir));
}

auto makeArray(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &values) -> std::array<amrex::Real, AMREX_SPACEDIM>
{
	std::array<amrex::Real, AMREX_SPACEDIM> result{};
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		result[d] = values[d];
	}
	return result;
}

auto makeInitializeHook(py::function func)
{
	return [func = std::move(func)](quokka::grid const &grid) {
		py::gil_scoped_acquire gil;
		func(makeGridObject(grid));
	};
}

auto makeExactHook(py::function func)
{
	return [func = std::move(func)](quokka::grid const &grid, amrex::Real time) {
		py::gil_scoped_acquire gil;
		func(makeGridObject(grid), time);
	};
}

auto makeDiagnosticsHook(py::function func)
{
	return [func = std::move(func)](AdvectionSimulationPrototype &sim) {
		py::gil_scoped_acquire gil;
		func(py::cast(&sim, py::return_value_policy::reference));
	};
}

void registerMultiFabExtensions(py::module_ const &amrex_module)
{
	auto ext_module = py::module_::import("pyquokka_py.extensions.multifab");
	auto register_func = ext_module.attr("register_multifab_extension");
	register_func(amrex_module);
}

auto makePyTuple(amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const &values) -> py::tuple
{
	py::tuple result(AMREX_SPACEDIM);
	for (int d = 0; d < AMREX_SPACEDIM; ++d) {
		result[d] = py::float_(values[d]);
	}
	return result;
}

} // namespace

PYBIND11_MODULE(pyquokka, m)
{
	auto importPyAmrexModule = []() { return py::module_::import(amrexPythonModuleName()); };
	auto ensurePyAmrexModuleReady = [importPyAmrexModule]() {
		auto amrex_module = importPyAmrexModule();
		// Just mark the module as initialized by pyquokka, but don't call its initialize()
		// since we're initializing AMReX directly below
		if (!py::hasattr(amrex_module, "_pyquokka_initialized") || !amrex_module.attr("_pyquokka_initialized").cast<bool>()) {
			amrex_module.attr("_pyquokka_initialized") = py::bool_(true);
		}
		return amrex_module;
	};

	m.def("initialize", [ensurePyAmrexModuleReady]() {
		if (!amrex::Initialized()) {
			int argc = 1;
			char arg0[] = "pyquokka";
			char *argv[] = {arg0, nullptr};
			char **argv_ptr = argv;
			amrex::Initialize(argc, argv_ptr);
		}
		ensurePyAmrexModuleReady();
	});
	m.def("finalize", [importPyAmrexModule]() {
		if (amrex::Initialized()) {
			amrex::Finalize();
		}
		auto amrex_module = importPyAmrexModule();
		if (py::hasattr(amrex_module, "_pyquokka_initialized")) {
			amrex_module.attr("_pyquokka_initialized") = py::bool_(false);
		}
	});

	ensureArray4Bindings(m);
	auto amrex_module = importPyAmrexModule();
	m.attr("amr") = amrex_module;
	registerMultiFabExtensions(amrex_module);

	py::class_<quokka::grid>(m, "Grid")
	    .def_property_readonly("dx", [](quokka::grid const &grid) { return makeArray(grid.dx_); })
	    .def_property_readonly("prob_lo", [](quokka::grid const &grid) { return makeArray(grid.prob_lo_); })
	    .def_property_readonly("prob_hi", [](quokka::grid const &grid) { return makeArray(grid.prob_hi_); })
	    .def_property_readonly("array4",
				   [](quokka::grid const &grid) { return py::cast(grid.array_, py::return_value_policy::reference); })
	    .def_property_readonly("i_range", [](quokka::grid const &grid) { return makeRangePair(grid.indexRange_, 0); })
#if (AMREX_SPACEDIM >= 2)
	    .def_property_readonly("j_range", [](quokka::grid const &grid) { return makeRangePair(grid.indexRange_, 1); })
#endif
#if (AMREX_SPACEDIM >= 3)
	    .def_property_readonly("k_range", [](quokka::grid const &grid) { return makeRangePair(grid.indexRange_, 2); })
#endif
	    .def("get_state",
		 [](quokka::grid const &grid, int i, int j, int k, int component) { return grid.array_(i, j, k, component); }, py::arg("i"),
		 py::arg("j") = 0, py::arg("k") = 0, py::arg("component") = 0)
	    .def("set_state",
		 [](quokka::grid const &grid, int i, int j, int k, int component, amrex::Real value) { grid.array_(i, j, k, component) = value; },
		 py::arg("i"), py::arg("j") = 0, py::arg("k") = 0, py::arg("component") = 0, py::arg("value"))
	    .def("set_state",
		 [](quokka::grid const &grid, int i, amrex::Real value) { grid.array_(i, 0, 0, 0) = value; }, py::arg("i"), py::arg("value"))
	    .def("get_state",
		 [](quokka::grid const &grid, int i) { return grid.array_(i, 0, 0, 0); }, py::arg("i"));

	py::class_<AdvectionSimulationPrototype>(m, "AdvectionSimulation")
	    .def(py::init<>())
	    .def("configure_grid", &AdvectionSimulationPrototype::configureGrid, py::arg("nx"), py::arg("prob_lo"),
		 py::arg("prob_hi"))
	    .def("set_advection_velocity", &AdvectionSimulationPrototype::setAdvectionVelocity, py::arg("vx"), py::arg("vy"),
		 py::arg("vz"))
	    .def("set_cfl", &AdvectionSimulationPrototype::setCfl)
	    .def("set_max_time", &AdvectionSimulationPrototype::setMaxTime)
	    .def("set_max_timesteps", &AdvectionSimulationPrototype::setMaxTimesteps)
	    .def("set_python_hooks",
		 [](AdvectionSimulationPrototype &sim, py::object init, py::object exact, py::object diagnostics) {
			 AdvectionSimulationPrototype::Callbacks callbacks;
			 if (!init.is_none()) {
				 callbacks.hooks.initialize = makeInitializeHook(py::function(init));
			 }
			 if (!exact.is_none()) {
				 callbacks.hooks.exact_solution = makeExactHook(py::function(exact));
			 }
			 if (!diagnostics.is_none()) {
				 callbacks.hooks.diagnostics = makeDiagnosticsHook(py::function(diagnostics));
			 }
			 sim.setCallbacks(std::move(callbacks));
		 },
		 py::arg("initial") = py::none(), py::arg("exact") = py::none(), py::arg("diagnostics") = py::none())
	    .def("run", &AdvectionSimulationPrototype::run)
	    .def("error_norm", &AdvectionSimulationPrototype::errorNorm)
	    .def("current_time", &AdvectionSimulationPrototype::currentTime)
	    .def(
		"state",
		[](py::object self, int level) -> py::object {
			auto &sim = self.cast<AdvectionSimulationPrototype &>();
			amrex::MultiFab &mf = sim.state(level);
			py::object mf_obj = py::cast(&mf, py::return_value_policy::reference, self);
			const auto &geom = sim.Geom(level);
			mf_obj.attr("_quokka_level") = level;
			mf_obj.attr("_quokka_prob_lo") = makePyTuple(geom.ProbLoArray());
			mf_obj.attr("_quokka_prob_hi") = makePyTuple(geom.ProbHiArray());
			mf_obj.attr("_quokka_dx") = makePyTuple(geom.CellSizeArray());
			return mf_obj;
		},
		py::arg("level") = 0,
		py::return_value_policy::reference,
		py::keep_alive<0, 1>());
}
