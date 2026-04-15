from collections import defaultdict

import numpy as np


# Define the structure of the entries for each parameter
def default_param():
    return {
        "name": "",
        "value": None,
        "unit": "",
        "descr": "",
        "group": "",
        "part": "",
        "type": "",
    }


# Create the dict for storing parameters of model
param = defaultdict(default_param)


# Define a function for unpacking some values
def unpack(name, parameter_set=param):
    """Get the values of a parameters saved in p
    Args:
        name (str or list of str): The name of a parameter or a list of such names
    Returns:
        float of list of float: If name is a str, the value of the parameter with that name, otherwise if name is a list get the values for the parameter names in that list
    """
    # Get the parameter value of name if it's a str
    if isinstance(name, str):
        return parameter_set.get(name).get("value")
    # Get the parameter values of the names in name if it's a list
    elif isinstance(name, list):
        return [parameter_set.get(x).get("value") for x in name]
    else:
        raise ValueError("name must be str or list of str")


def make_valuedict(parameter_set: dict, type_select: "list|str" = None):
    if type_select is None:
        return {key: value.get("value") for key, value in parameter_set.items()}
    elif isinstance(type_select, str):
        return {
            key: value.get("value")
            for key, value in parameter_set.items()
            if value.get("type") == type_select
        }
    elif isinstance(type_select, list) and np.all([type(x) is str for x in type_select]):
        return {
            key: value.get("value")
            for key, value in parameter_set.items()
            if value.get("type") in type_select
        }
    else:
        raise ValueError("type select must be of type Str or List[Str]")


# Stimulation parameters
param["stim_start"] = {
    "name": "",
    "value": 10.0,
    "unit": "",
    "descr": "",
    "group": "Stimulation parameters",
    "part": "",
    "type": "parameter",
}
param["stim_period"] = {
    "name": "",
    "value": 1000.0,
    "unit": "",
    "descr": "",
    "group": "Stimulation parameters",
    "part": "",
    "type": "parameter",
}
param["stim_duration"] = {
    "name": "",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Stimulation parameters",
    "part": "",
    "type": "parameter",
}
param["stim_amplitude"] = {
    "name": "",
    "value": 52.0,
    "unit": "",
    "descr": "",
    "group": "Stimulation parameters",
    "part": "",
    "type": "parameter",
}

# Cell parameters
param["T"] = {
    "name": "Temperature",
    "value": 310.0,
    "unit": "kelvin",
    "descr": "",
    "group": "Cell parameters",
    "part": "in membrane",
    "type": "parameter",
}
param["F"] = {
    "name": "",
    "value": 96485.3415,
    "unit": "coulomb_per_millimole",
    "descr": "",
    "group": "Cell parameters",
    "part": "in membrane",
    "type": "parameter",
}
param["R"] = {
    "name": "",
    "value": 8314.472,
    "unit": "joule_per_mole_kelvin",
    "descr": "",
    "group": "Cell parameters",
    "part": "in membrane",
    "type": "parameter",
}
param["V_c"] = {
    "name": "Vc",
    "value": 0.016404,
    "unit": "micrometre^3",
    "descr": "",
    "group": "Cell parameters",
    "part": "in membrane",
    "type": "parameter",
}
param["V_sr"] = {
    "name": "VSR",
    "value": 0.001094,
    "unit": "micrometre^3",
    "descr": "",
    "group": "Cell parameters",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["V_ss"] = {
    "name": "Vss",
    "value": 5.47e-05,
    "unit": "micrometre^3",
    "descr": "",
    "group": "Cell parameters",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Ca_o"] = {
    "name": "[Ca]o",
    "value": 2.0,
    "unit": "millimolar",
    "descr": "",
    "group": "Cell parameters",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Na_o"] = {
    "name": "[Na]o",
    "value": 140.0,
    "unit": "millimolar",
    "descr": "",
    "group": "Cell parameters",
    "part": "in sodium_dynamics",
    "type": "parameter",
}
param["K_o"] = {
    "name": "[K]o",
    "value": 5.4,
    "unit": "millimolar",
    "descr": "",
    "group": "Cell parameters",
    "part": "in potassium_dynamics",
    "type": "parameter",
}
param["Cm"] = {
    "name": "Capacitance",
    "value": 0.185,
    "unit": "microF",
    "descr": "",
    "group": "Cell parameters",
    "part": "in membrane",
    "type": "parameter",
}

# Parameters for the Ca handling
param["k_o_Ca"] = {
    "name": "K1_prime",
    "value": 2.1,
    "unit": "per_millimolar^2_per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["k_i_Ca"] = {
    "name": "K2_prime",
    "value": 0.025,
    "unit": "per_millimolar_per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["k_om"] = {
    "name": "k3",
    "value": 0.06,
    "unit": "per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["k_im"] = {
    "name": "k4",
    "value": 0.005,
    "unit": "per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["K_up"] = {
    "name": "Kup",
    "value": 0.00025,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["max_sr"] = {
    "name": "max_SR",
    "value": 2.5,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["min_sr"] = {
    "name": "min_SR",
    "value": 1.0,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["EC"] = {
    "name": "EC",
    "value": 1.5,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["V_leak"] = {
    "name": "Vleak",
    "value": 0.00036,
    "unit": "per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["V_rel"] = {
    "name": "Vrel",
    "value": 2.5, #1.5
    "unit": "per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["V_xfer"] = {
    "name": "Vxfer",
    "value": 0.00456,
    "unit": "per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Vmax_up"] = {
    "name": "Vmaxup",
    "value": 0.00058, # 0.00765 0.00068 88
    "unit": "millimolar_per_millisecond",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Buf_c"] = {
    "name": "Bufc",
    "value": 0.11,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Buf_sr"] = {
    "name": "BufSR",
    "value": 10.0,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["Buf_ss"] = {
    "name": "Bufss",
    "value": 0.4,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["K_buf_c"] = {
    "name": "Kbuffc",
    "value": 0.00085,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["K_buf_sr"] = {
    "name": "KbuffSR",
    "value": 0.3,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["K_buf_ss"] = {
    "name": "KbuffSS",
    "value": 0.00025,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in calcium_dynamics",
    "type": "parameter",
}
param["A_tot"] = {
    "name": "TnС общий",
    "value": 0.07,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in intracellular_calcium_concentration",
    "type": "parameter",
}
param["a_on"] = {
    "name": "Константа связывания TnС",
    "value": 36.0,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in intracellular_calcium_concentration",
    "type": "parameter",
}
param["a_off"] = {
    "name": "Константа распада TnС",
    "value": 0.19,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in intracellular_calcium_concentration",
    "type": "parameter",
}
param["k_A"] = {
    "name": "qa",
    "value": 28.0,
    "unit": "per_millimolar",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["kdecay"] = {  # такого параметра у меня нет и он нигде в Наташи не используется
    "name": "---",
    "value": 10.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "",
    "type": "parameter",
}
param["K_inh"] = {
    "name": "kinh",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the Ca handling",
    "part": "",
    "type": "parameter",
}

# Parameters for the mechanical part

param["alpha_1"] = {
    "name": "alfa1",
    "value": 14.6,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["beta_1"] = {
    "name": "beta1",
    "value": 4.2,
    "unit": "millinewton",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["alpha_2"] = {
    "name": "alfa2",
    "value": 14.6,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["beta_2"] = {
    "name": "beta2",
    "value": 0.009,
    "unit": "millinewton",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["llambda"] = {
    "name": "lambda",
    "value": 450.0, # 450
    "unit": "millinewton",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["q_1"] = {
    "name": "q1",
    "value": 0.0173,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["q_2"] = {
    "name": "q2",
    "value": 0.259,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["q_3"] = {
    "name": "q3",
    "value": 0.0173,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["q_4"] = {
    "name": "q4",
    "value": 0.015,
    "unit": "per_second",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["v_max"] = {
    "name": "vmax",
    "value": 0.0055,
    "unit": "micrometre_per_second",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}  # {скорость укорочения при нулевой нагрузке}
param["a"] = {
    "name": "a",
    "value": 0.25,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["alpha_Q"] = {
    "name": "alfaq",
    "value": 10.0,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["beta_Q"] = {
    "name": "betaq",
    "value": 5.0,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["x_st"] = {
    "name": "Xst",
    "value": 0.964285,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}  # {Хst=Vst/Vmax, Vst-скорость при которой резко уменьшается qn}
param["alpha_G"] = {
    "name": "alfag",
    "value": 1.0,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["m_0"] = {
    "name": "m0",
    "value": 0.9,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}  # {начальная величина вероятности прикрепления n2}
param["g_1"] = {
    "name": "g1",
    "value": 0.6,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in crossbridge_kinetics",
    "type": "parameter",
}
param["g_2"] = {
    "name": "g2",
    "value": 0.52,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in crossbridge_kinetics",
    "type": "parameter",
}
param["s_0"] = {
    "name": "s0",
    "value": 1.14,
    "unit": "micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters_izakov_et_al_1991",
    "type": "parameter",
}
param["s046"] = {
    "name": "s046",
    "value": 0.46,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["s055"] = {
    "name": "s055",
    "value": 0.55,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["chi_1"] = {
    "name": "каппа1",
    "value": 0.55, # 0.5
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}  # {изменение циклирования мостика}
param["pi_min"] = {
    "name": "pimin",
    "value": 0.02,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["r0"] = {
    "name": "ro",
    "value": 2.55248904424517,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["d_h"] = {
    "name": "d",
    "value": 0.5,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["v_1"] = {  # проверить что за параметр
    "name": "x1",
    "value": 0.1,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["x0"] = {  # проверить что за параметр
    "name": "x0",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["m"] = {
    "name": "m",
    "value": 1.7,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["chi_0"] = {
    "name": "каппа0",
    "value": 2.1,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["alpha_P"] = {
    "name": "alphap",
    "value": 4.0,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["q_st"] = {
    "name": "qst",
    "value": 1000.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["alpha_3"] = {
    "name": "alpha3",
    "value": 55.0,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["beta_3"] = {
    "name": "beta3",
    "value": 0.11,
    "unit": "millinewton",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["alpha_tir"] = {  # Проверить, что за параметр
    "name": "alpha_tir",
    "value": 30.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["beta_tir"] = {  # Проверить, что за параметр
    "name": "beta_tir",
    "value": 0.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["l_tir"] = {  # Проверить, что за параметр
    "name": "l_tir",
    "value": -0.03,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["beta_vp_l"] = {
    "name": "vs1",
    "value": 0.1,
    "unit": "millinewton_second_per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in CE_velocity",
    "type": "parameter",
}
param["alpha_vp_l"] = {
    "name": "alp_vp",
    "value": 16.0,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in CE_velocity",
    "type": "parameter",
}
param["beta_vp_s"] = {
    "name": "vs1rel",
    "value": 10.0,
    "unit": "millinewton_second_per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in CE_velocity",
    "type": "parameter",
}
param["alpha_vp_s"] = {
    "name": "alp_vpr",
    "value": 16.0,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in CE_velocity",
    "type": "parameter",
}
param["beta_vs_l"] = {
    "name": "vs2",
    "value": 20.0,
    "unit": "millinewton_second_per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in PE_velocity",
    "type": "parameter",
}
param["alpha_vs_l"] = {
    "name": "alp_vs",
    "value": 46.0,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in PE_velocity",
    "type": "parameter",
}
param["beta_vs_s"] = {
    "name": "vs2rel",
    "value": 60.0,
    "unit": "millinewton_second_per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in PE_velocity",
    "type": "parameter",
}
param["alpha_vs_s"] = {
    "name": "alp_vsr",
    "value": 39.0,
    "unit": "per_micrometre",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in PE_velocity",
    "type": "parameter",
}
param["s_c"] = {
    "name": "sc",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["k_mu"] = {
    "name": "k_mu",
    "value": 0.6,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["mu"] = {
    "name": "mu1",
    "value": 3.3,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "in parameters",
    "type": "parameter",
}
param["chi_2"] = {
    "name": "kappa2",
    "value": 0.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_B"] = {
    "name": "Bl1",
    "value": 55.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_Q"] = {
    "name": "Ql1",
    "value": 0.835,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_K"] = {
    "name": "Kl1",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_A"] = {
    "name": "Al1",
    "value": 0.5,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_C"] = {
    "name": "Cl1",
    "value": 1.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}
param["n1_nu"] = {
    "name": "nul1",
    "value": 5.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the mechanical part",
    "part": "",
    "type": "parameter",
}

# Parameters for the electrical part

param["g_Kr"] = {
    "name": "gkr",
    "value": 0.153,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in rapid_time_dependent_potassium_current",
    "type": "parameter",
}
param["g_pK"] = {
    "name": "gpk",
    "value": 0.0146,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in potassium_pump_current",
    "type": "parameter",
}
param["g_Ks"] = {
    "name": "gks",
    "value": 0.392,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in slow_time_dependent_potassium_current",
    "type": "parameter",
}
param["g_K1"] = {
    "name": "gK1",
    "value": 5.405,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in inward_rectifier_potassium_current",
    "type": "parameter",
}
param["P_NaK"] = {
    "name": "pNaK",
    "value": 2.724,
    "unit": "picoA_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_potassium_pump_current",
    "type": "parameter",
}
param["P_kna"] = {
    "name": "pKNa",
    "value": 0.03,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in reversal_potentials",
    "type": "parameter",
}
param["K_mk"] = {
    "name": "kmK",
    "value": 1.0,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_potassium_pump_current",
    "type": "parameter",
}
param["K_mNa"] = {
    "name": "kmNa",
    "value": 40.0,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_potassium_pump_current",
    "type": "parameter",
}
param["g_to"] = {
    "name": "gto",
    "value": 0.735, # 0.515
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in transient_outward_current",
    "type": "parameter",
}
param["g_Na"] = {
    "name": "gNa",
    "value": 14.838,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in fast_sodium_current",
    "type": "parameter",
}
param["g_bna"] = {
    "name": "gbNa",
    "value": 0.00029,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_background_current",
    "type": "parameter",
}
param["alpha"] = {
    "name": "alpha_NaCa",
    "value": 1.0, # 1.5
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["gamma"] = {
    "name": "gamma_NaCa",
    "value": 0.35,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["K_NaCa"] = {
    "name": "kNaCa",
    "value": 5000.0, # 5000.0
    "unit": "picoA_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["K_sat"] = {
    "name": "Ksat",
    "value": 0.1,
    "unit": "dimensionless",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["Km_Ca"] = {
    "name": "KmCa",
    "value": 1.38,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["Km_Nai"] = {
    "name": "KmNai",
    "value": 87.5,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in sodium_calcium_exchanger_current",
    "type": "parameter",
}
param["g_bCa"] = {
    "name": "gbCa",
    "value": 0.000592,
    "unit": "nanoS_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in calcium_background_current",
    "type": "parameter",
}
param["g_pCa"] = {
    "name": "gpCa",
    "value": 0.2476,
    "unit": "picoA_per_picoF",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in calcium_pump_current",
    "type": "parameter",
}
param["K_pCa"] = {
    "name": "KpCa",
    "value": 0.0005,
    "unit": "millimolar",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in calcium_pump_current",
    "type": "parameter",
}
param["g_CaL"] = {
    "name": "gCaL",
    "value": 5.0e-5,
    "unit": "litre_per_farad_second",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "in L_type_Ca_current",
    "type": "parameter",
}
param["Ca_sense"] = {
    "name": "Casense",
    "value": 0.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "",
    "type": "parameter",
}
param["k_Ca_sense"] = {
    "name": "kCasense",
    "value": 0.0069,
    "unit": "",
    "descr": "",
    "group": "Parameters for the electrical part",
    "part": "",
    "type": "parameter",
}

# Parameters for ischemic params

param["[ATP]i"] = {
    "name": "[ATP]i",
    "value": 6.8,
    "unit": "",
    "descr": "",
    "group": "Parameters for ischemic params",
    "part": "",
    "type": "parameter",
}
param["KmATP"] = {
    "name": "KmATP",
    "value": 0.0976,
    "unit": "",
    "descr": "",
    "group": "Parameters for ischemic params",
    "part": "",
    "type": "parameter",
}
param["gKATP"] = {
    "name": "gKATP",
    "value": 1.59294,
    "unit": "",
    "descr": "",
    "group": "Parameters for ischemic params",
    "part": "",
    "type": "parameter",
}
param["Ko,norm"] = {
    "name": "Ko,norm",
    "value": 4.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for ischemic params",
    "part": "",
    "type": "parameter",
}

# parameters for CaMK part

param["kp"] = {
    "name": "kp",
    "value": 1000.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part (Semin)",
    "part": "",
    "type": "parameter",
}
param["KP"] = {
    "name": "KP",
    "value": 0.000325,
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part (Semin)",
    "part": "",
    "type": "parameter",
}

param["V_nSR"] = {
    "name": "V_nSR",
    "value": 0.00100648,
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part",
    "part": "",
    "type": "parameter",
}

param["V_jSR"] = {
    "name": "V_jSR",
    "value": 0.00008752,
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part",
    "part": "",
    "type": "parameter",
}

param["K_buf_jsr"] = {
    "name": "K_buf_jsr",
    "value": 0.3,  # 0.3 0.8
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part",
    "part": "",
    "type": "parameter",
}

param["Buf_jsr"] = {
    "name": "Buf_jsr",
    "value": 10.0,
    "unit": "",
    "descr": "",
    "group": "Parameters for CaMK part",
    "part": "",
    "type": "parameter",
}

# Phase variables

param["Cai"] = {
    "name": "Cai",
    "value": 4.34295055e-05,
    "unit": "millimolar",
    "descr": "",
    "group": "Phase variables",
    "part": "in calcium_dynamics",
    "type": "initial_conditions",
}
param["CaTn"] = {
    "name": "CaTn",
    "value": 6.31929074e-04,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["CaSS"] = {
    "name": "CaSS",
    "value": 1.57457522e-04,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["CaSR"] = {
    "name": "CaSR",
    "value": 9.28218670e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["l1"] = {
    "name": "l1",
    "value": 3.86134451e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["l2"] = {
    "name": "l2",
    "value": 3.86910662e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["N"] = {
    "name": "N",
    "value": 1.34966357e-06,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["l3"] = {
    "name": "l3",
    "value": 5.80515391e-02,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["v"] = {
    "name": "v",
    "value": 3.39882644e-06,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["w"] = {
    "name": "w",
    "value": 8.52399973e-07,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

param["xr1"] = {
    "name": "xr1",
    "value": 1.92071800e-04,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["xr2"] = {
    "name": "xr2",
    "value": 4.78422683e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["xs"] = {
    "name": "xs",
    "value": 3.09815853e-03,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["r"] = {
    "name": "r",
    "value": 2.15152212e-08,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["s"] = {
    "name": "s",
    "value": 9.99998122e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["m"] = {
    "name": "m",
    "value": 1.47900595e-03,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["h"] = {
    "name": "h",
    "value": 7.63521043e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["j"] = {
    "name": "j",
    "value": 7.62863363e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["d"] = {
    "name": "d",
    "value": 3.07278916e-05,
    "unit": "dimensionless",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["f"] = {
    "name": "f",
    "value": 9.82107417e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["f2"] = {
    "name": "f2",
    "value": 9.99476773e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["fCaSS"] = {
    "name": "fCaSS",
    "value": 9.99970647e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["Nai"] = {
    "name": "Nai",
    "value": 1.02371819e+01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["Ki"] = {
    "name": "Ki",
    "value": 1.35877968e+02 ,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["E"] = {
    "name": "E",
    "value": -8.59273503e+01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["R_RyR"] = {
    "name": "R_RyR",
    "value": 9.88267582e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["O_RyR"] = {
    "name": "O_RyR",
    "value": 4.23457956e-07,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["I_RyR"] = {
    "name": "I_RyR",
    "value": 5.02698275e-09,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}
param["RI_RyR"] = {
    "name": "RI_RyR",
    "value": 1.17319890e-02,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

param["p_iup"] = {
    "name": "p_iup",
    "value": 4.88823882e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

param["CaMKt"] = {
    "name": "CaMKt",
    "value": 0.0110752904836162,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

param["l0"] = {
    "name": "l0",
    "value": unpack("l2") + unpack("l3"),
    "unit": "",
    "descr": "",
    "group": "",
    "part": "",
    "type": "",
}

param["Ca_nSR"] = {
    "name": "Ca_nSR",
    "value": 4.28218670e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

param["Ca_jSR"] = {
    "name": "Ca_jSR",
    "value": 4.88218670e-01,
    "unit": "",
    "descr": "",
    "group": "Phase variables",
    "part": "",
    "type": "initial_conditions",
}

###################################################
# Check if the current parameters are up to date
###################################################


def _make_readable_value(x, num=False):
    if np.abs(x) > 1000 or np.abs(x) < 0.0001:
        res = "%.3e" % x
    else:
        res = "%.3f" % x

    if num:
        return float(res)
    else:
        return res


def make_readable_value(x, num=False):
    if isinstance(x, (int, float)):
        return _make_readable_value(x, num)
    elif isinstance(x, dict):
        res = {k: _make_readable_value(v, num) for k, v in x}
    else:
        res = np.zeros(len(x))

        for i, sub_val in enumerate(x):
            res[i] = _make_readable_value(sub_val, num)

        return res


def check_parameters(p_check):
    p_new = {
        key: make_readable_value(param[key]["value"], True)
        for key in p_check.keys()
        if key not in ["pfd"]
    }

    if p_check == p_new:
        return True
    else:
        print("given dict | here")
        for key, value in p_new.items():
            vals_old = make_readable_value(param[key]["value"], False)
            vals_new = make_readable_value(value, False)
            if np.any(vals_old != vals_new):
                print("%s: %s | %s" % (key, vals_old, vals_new))


###########################
# Print parameter values
###########################


def _export_num(key, value, sep):
    return f"    '{key}': {str(make_readable_value(value))}{sep}"


def _exportiter_list(value):
    return [str(make_readable_value(v)) for i, v in enumerate(value)]


def _exportiter_dict(value):
    return [f"'{k}': {str(make_readable_value(v))}" for i, (k, v) in enumerate(value.items())]


def _export_container(key, value, sep, iterfun, parenteses):
    prnt = f"    '{key}': {parenteses[0]}"

    items = iterfun(value)
    prnt += ", ".join(items)
    prnt += parenteses[1] + sep
    return prnt


def _export_list(key, value, sep):
    return _export_container(key, value, sep, _exportiter_list, ["[", "]"])


def _export_dict(key, value, sep):
    return _export_container(key, value, sep, _exportiter_dict, ["{", "}"])


def _export_pdSeries(key, value, sep):
    return _export_container(key, value.to_dict(), sep, _exportiter_dict, ["pd.Series({", "})"])


def export_parameters_str(p_list, sep=",", annotate=True, include_parentheses=True):
    res = ""
    for key, value_dict in p_list.items():
        value = value_dict["value"]

        if isinstance(value, (int, float)):
            prnt = _export_num(key, value, sep)
        else:
            raise ValueError(
                f"value of parameter {key} has type {type(value)} which has no export function"
            )
        if annotate:
            prnt += f" # {value_dict['name']} [{value_dict['unit']}] {value_dict['descr']} ({value_dict['group']}) ({value_dict['part']}) - {value_dict['type']}"
        res += prnt + "\n"
    if include_parentheses:
        res = "{\n" + res + "}"
    return res


def print_parameters(p_list, sep=",", annotate=True):
    print(export_parameters_str(p_list, sep=sep, annotate=annotate, include_parentheses=False))
    print("")


def get_parameters(p_list):
    res = {}
    for key, value_dict in p_list.items():
        value = value_dict["value"]
        res[key] = make_readable_value(value, num=True)

    return res


if __name__ == "__main__":
    print("---- Model parameters ----")
    print_parameters(param)
    print("\n")
