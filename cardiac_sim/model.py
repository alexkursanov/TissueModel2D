"""Модель TNNPM: электромеханическое сопряжение кардиомиоцита.

Объединяет:
- электрофизиологию TNNP (Ten Tusscher-Noble-Noble-Panfilov),
- кальциевую динамику с 4-состоячными RyR-каналами (Shannon),
- механику саркомера (Екатеринбургская модель).

Вектор состояния (33 переменные):
    [0]  d       — ворота L-type Ca²⁺ канала (активация)
    [1]  f2      — ворота L-type Ca²⁺ канала (инактивация 2)
    [2]  fCass   — ворота L-type Ca²⁺ канала (инактивация Ca-зависимая)
    [3]  f       — ворота L-type Ca²⁺ канала (инактивация 1)
    [4]  Ca_SR   — Ca²⁺ в SR (устаревшая переменная, dCa_SR = 0)
    [5]  Ca_i    — внутриклеточный Ca²⁺
    [6]  Ca_ss   — Ca²⁺ в субпространстве
    [7]  p_iup   — фосфорилирование SERCA (Semin)
    [8]  h       — ворота Na⁺ канала (быстрая инактивация)
    [9]  j       — ворота Na⁺ канала (медленная инактивация)
    [10] m       — ворота Na⁺ канала (активация)
    [11] V       — трансмембранный потенциал (мВ)
    [12] K_i     — внутриклеточный K⁺
    [13] Xr1     — ворота i_Kr (активация)
    [14] Xr2     — ворота i_Kr (инактивация)
    [15] Xs      — ворота i_Ks
    [16] Na_i    — внутриклеточный Na⁺
    [17] r       — ворота i_to (активация)
    [18] s       — ворота i_to (инактивация)
    [19] v       — скорость CE
    [20] w       — скорость PE
    [21] N       — доля прикреплённых поперечных мостиков
    [22] A       — Ca²⁺-связанный TnC
    [23] l_1     — длина PE элемента
    [24] l_2     — длина SE элемента
    [25] l_3     — длина XSE элемента
    [26] R       — RyR: состояние покоя
    [27] O       — RyR: открытое
    [28] I       — RyR: инактивированное
    [29] RI      — RyR: RI
    [30] CaMKt   — активность CaMKII
    [31] Ca_nSR  — Ca²⁺ в нетканевом SR
    [32] Ca_jSR  — Ca²⁺ в юнкциональном SR
"""

from numpy import exp, floor, log, sqrt

import numpy as np

from .calculate_parameters import param


class TNNPM:
    """Модель электромеханического сопряжения кардиомиоцита (TNNP + EKB + Shannon)."""

    def __init__(self) -> None:
        self.forces: np.ndarray = np.zeros(6, dtype=np.float64)
        self.currents: np.ndarray = np.zeros(12, dtype=np.float64)
        self.F_afterload: float = 0.0

        # Механические параметры
        self.l0: float = param["l0"]["value"]
        self.r0: float = param["r0"]["value"]
        self.llambda: float = param["llambda"]["value"]
        self.alpha_vp_l: float = param["alpha_vp_l"]["value"]
        self.beta_vp_l: float = param["beta_vp_l"]["value"]
        self.alpha_vp_s: float = param["alpha_vp_s"]["value"]
        self.beta_vp_s: float = param["beta_vp_s"]["value"]
        self.alpha_vs_l: float = param["alpha_vs_l"]["value"]
        self.beta_vs_l: float = param["beta_vs_l"]["value"]
        self.alpha_vs_s: float = param["alpha_vs_s"]["value"]
        self.beta_vs_s: float = param["beta_vs_s"]["value"]
        self.alpha_1: float = param["alpha_1"]["value"]
        self.beta_1: float = param["beta_1"]["value"]
        self.alpha_2: float = param["alpha_2"]["value"]
        self.beta_2: float = param["beta_2"]["value"]
        self.alpha_3: float = param["alpha_3"]["value"]
        self.beta_3: float = param["beta_3"]["value"]

        # N (поперечные мостики)
        self.m_0: float = param["m_0"]["value"]
        self.chi_0: float = param["chi_0"]["value"]
        self.chi_1: float = param["chi_1"]["value"]
        self.chi_2: float = param["chi_2"]["value"]
        self.v_max: float = param["v_max"]["value"]

        # q-функция
        self.q_1: float = param["q_1"]["value"]
        self.q_2: float = param["q_2"]["value"]
        self.q_3: float = param["q_3"]["value"]
        self.q_4: float = param["q_4"]["value"]
        self.x_st: float = param["x_st"]["value"]
        self.alpha_Q: float = param["alpha_Q"]["value"]
        self.beta_Q: float = param["beta_Q"]["value"]

        # G_star
        self.a: float = param["a"]["value"]
        self.v_1: float = param["v_1"]["value"]
        self.alpha_G: float = param["alpha_G"]["value"]
        self.alpha_P: float = param["alpha_P"]["value"]

        # P_star
        self.d_h: float = param["d_h"]["value"]

        # M (кооперативность TnC)
        self.A_tot: float = param["A_tot"]["value"]
        self.mu: float = param["mu"]["value"]
        self.k_mu: float = param["k_mu"]["value"]

        # n_1
        self.g_1: float = param["g_1"]["value"]
        self.g_2: float = param["g_2"]["value"]
        self.n1_A: float = param["n1_A"]["value"]
        self.n1_B: float = param["n1_B"]["value"]
        self.n1_C: float = param["n1_C"]["value"]
        self.n1_K: float = param["n1_K"]["value"]
        self.n1_Q: float = param["n1_Q"]["value"]
        self.n1_nu: float = param["n1_nu"]["value"]

        # L_oz (перекрытие актина и миозина)
        self.s_0: float = param["s_0"]["value"]
        self.s046: float = param["s046"]["value"]
        self.s055: float = param["s055"]["value"]

        # dA (связывание Ca²⁺ с TnC)
        self.a_on: float = param["a_on"]["value"]
        self.a_off: float = param["a_off"]["value"]
        self.k_A: float = param["k_A"]["value"]

        # pi_N_A
        self.s_c: float = param["s_c"]["value"]
        self.pi_min: float = param["pi_min"]["value"]

        # Стимуляция
        self.stim_start: float = param["stim_start"]["value"]
        self.stim_period: float = param["stim_period"]["value"]
        self.stim_duration: float = param["stim_duration"]["value"]
        self.stim_amplitude: float = param["stim_amplitude"]["value"]

        # Внешние концентрации ионов
        self.Ca_o: float = param["Ca_o"]["value"]
        self.K_o: float = param["K_o"]["value"]
        self.Na_o: float = param["Na_o"]["value"]

        # Физические константы и параметры клетки
        self.Cm: float = param["Cm"]["value"]
        self.F: float = param["F"]["value"]
        self.T: float = param["T"]["value"]
        self.R: float = param["R"]["value"]

        # Буферизация Ca²⁺ в SR
        self.Buf_sr: float = param["Buf_sr"]["value"]
        self.K_buf_sr: float = param["K_buf_sr"]["value"]

        # SERCA (Semin)
        self.k_sm_p: float = param["kp"]["value"]
        self.K_lrg_p: float = param["KP"]["value"]

        # i_up / SERCA
        self.Vmax_up: float = param["Vmax_up"]["value"]
        self.K_up: float = param["K_up"]["value"]

        # i_leak
        self.V_leak: float = param["V_leak"]["value"]

        # RyR Shannon
        self.max_sr: float = param["max_sr"]["value"]
        self.min_sr: float = param["min_sr"]["value"]
        self.EC: float = param["EC"]["value"]
        self.k_o_Ca: float = param["k_o_Ca"]["value"]
        self.k_i_Ca: float = param["k_i_Ca"]["value"]
        self.k_om: float = param["k_om"]["value"]
        self.k_im: float = param["k_im"]["value"]

        # i_rel / i_xfer
        self.V_rel: float = param["V_rel"]["value"]
        self.V_xfer: float = param["V_xfer"]["value"]

        # Буферизация Ca²⁺ в цитоплазме
        self.Buf_c: float = param["Buf_c"]["value"]
        self.K_buf_c: float = param["K_buf_c"]["value"]
        self.V_sr: float = param["V_sr"]["value"]
        self.V_c: float = param["V_c"]["value"]

        # i_bCa
        self.g_bCa: float = param["g_bCa"]["value"]

        # i_pCa
        self.g_pCa: float = param["g_pCa"]["value"]
        self.K_pCa: float = param["K_pCa"]["value"]

        # i_NaCa
        self.gamma: float = param["gamma"]["value"]
        self.alpha: float = param["alpha"]["value"]
        self.K_sat: float = param["K_sat"]["value"]
        self.Km_Ca: float = param["Km_Ca"]["value"]
        self.Km_Nai: float = param["Km_Nai"]["value"]
        self.K_NaCa: float = param["K_NaCa"]["value"]

        # Буферизация Ca²⁺ в субпространстве
        self.Buf_ss: float = param["Buf_ss"]["value"]
        self.K_buf_ss: float = param["K_buf_ss"]["value"]
        self.V_ss: float = param["V_ss"]["value"]

        # Ионные токи
        self.g_CaL: float = param["g_CaL"]["value"]
        self.g_Na: float = param["g_Na"]["value"]
        self.g_K1: float = param["g_K1"]["value"]
        self.g_to: float = param["g_to"]["value"]
        self.g_Kr: float = param["g_Kr"]["value"]
        self.g_Ks: float = param["g_Ks"]["value"]
        self.P_kna: float = param["P_kna"]["value"]
        self.g_K_ATP: float = param["gKATP"]["value"]
        self.ATPi: float = param["[ATP]i"]["value"]
        self.KmATP: float = param["KmATP"]["value"]
        self.P_NaK: float = param["P_NaK"]["value"]
        self.K_mk: float = param["K_mk"]["value"]
        self.K_mNa: float = param["K_mNa"]["value"]
        self.g_bna: float = param["g_bna"]["value"]
        self.g_pK: float = param["g_pK"]["value"]

        # Объёмы SR
        self.V_nSR: float = param["V_nSR"]["value"]
        self.V_jSR: float = param["V_jSR"]["value"]
        self.K_buf_jsr: float = param["K_buf_jsr"]["value"]
        self.Buf_jsr: float = param["Buf_jsr"]["value"]

        # Начальные условия
        self.d_init: float = param["d"]["value"]
        self.f2_init: float = param["f2"]["value"]
        self.fCass_init: float = param["fCaSS"]["value"]
        self.f_init: float = param["f"]["value"]
        self.Ca_SR_init: float = param["CaSR"]["value"]
        self.Ca_i_init: float = param["Cai"]["value"]
        self.Ca_ss_init: float = param["CaSS"]["value"]
        self.p_iup: float = param["p_iup"]["value"]
        self.h_init: float = param["h"]["value"]
        self.j_init: float = param["j"]["value"]
        self.m_init: float = param["m"]["value"]
        self.V_init: float = param["E"]["value"]
        self.K_i_init: float = param["Ki"]["value"]
        self.Xr1_init: float = param["xr1"]["value"]
        self.Xr2_init: float = param["xr2"]["value"]
        self.Xs_init: float = param["xs"]["value"]
        self.Na_i_init: float = param["Nai"]["value"]
        self.r_init: float = param["r"]["value"]
        self.s_init: float = param["s"]["value"]
        self.v_init: float = param["v"]["value"]
        self.w_init: float = param["w"]["value"]
        self.N_init: float = param["N"]["value"]
        self.A_init: float = param["CaTn"]["value"]
        self.l_1_init: float = param["l1"]["value"]
        self.l_2_init: float = param["l2"]["value"]
        self.l_3_init: float = param["l3"]["value"]
        self.R_init: float = param["R_RyR"]["value"]
        self.O_init: float = param["O_RyR"]["value"]
        self.I_init: float = param["I_RyR"]["value"]
        self.RI_init: float = param["RI_RyR"]["value"]
        self.CaMKt_init: float = param["CaMKt"]["value"]
        self.Ca_nSR_init: float = param["Ca_nSR"]["value"]
        self.Ca_jSR_init: float = param["Ca_jSR"]["value"]

    # ------------------------------------------------------------------
    # Правая часть ОДУ
    # ------------------------------------------------------------------

    def diff_equations(self, time: float, state_variables: np.ndarray) -> np.ndarray:
        """Вычислить производные всех 33 переменных состояния.

        Побочный эффект: обновляет ``self.forces`` и ``self.currents``
        для постобработки через ``calculate_vars``.

        Args:
            time: текущее время (мс).
            state_variables: вектор состояния длиной 33.

        Returns:
            Массив производных длиной 33.
        """
        sv = np.float128(state_variables)

        # Распаковка вектора состояния
        d, f2, fCass, f = sv[0], sv[1], sv[2], sv[3]
        Ca_SR, Ca_i, Ca_ss, p_iup = sv[4], sv[5], sv[6], sv[7]
        h, j, m, V = sv[8], sv[9], sv[10], sv[11]
        K_i, Xr1, Xr2, Xs = sv[12], sv[13], sv[14], sv[15]
        Na_i, r, s = sv[16], sv[17], sv[18]
        v, w, N, A = sv[19], sv[20], sv[21], sv[22]
        l_1, l_2, l_3 = sv[23], sv[24], sv[25]
        R, O, I, RI = sv[26], sv[27], sv[28], sv[29]
        CaMKt = sv[30]
        Ca_nSR, Ca_jSR = sv[31], sv[32]

        # --- Кальциевая динамика и CaMK ---
        (
            dCaMKt, i_up, i_leak, i_tr, i_rel, i_xfer,
            dCa_nSR, dCa_jSR, dCa_SR, dCa_i, dCa_ss,
            i_CaL, dp_iup, dR, dO, dI, dRI,
        ) = self._calcium_subsystem(
            Ca_i, Ca_ss, Ca_jSR, Ca_nSR, Ca_SR,
            d, f, f2, fCass, V, Na_i, CaMKt, p_iup, N, A, R, O, I, RI,
        )

        # --- Ионные токи и ворота ---
        (
            i_Na, i_NaCa, i_NaK, i_K1, i_to, i_Kr,
            i_Ks, i_K_ATP, i_b_Na, i_b_Ca, i_p_Ca, i_p_K,
            dd, df2, dfCass, df, dh, dj, dm,
            dXr1, dXr2, dXs, dr, ds,
        ) = self._electrophysiology_subsystem(
            V, d, f2, fCass, f, h, j, m, Xr1, Xr2, Xs, r, s,
            K_i, Na_i, Ca_i, Ca_ss,
        )

        # --- Мембранный потенциал и ионные концентрации ---
        dV, dK_i, dNa_i = self._membrane_subsystem(
            time, V, K_i, Na_i,
            i_K1, i_to, i_Kr, i_Ks, i_CaL, i_NaK, i_Na,
            i_b_Na, i_NaCa, i_b_Ca, i_p_K, i_p_Ca, i_K_ATP,
        )

        # --- Механика ---
        dN, dA, dv, dw, dl_1, dl_2, dl_3, forces = self._mechanics_subsystem(
            v, w, N, A, l_1, l_2, l_3, Ca_i,
        )
        F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE = forces

        # Сохраняем вспомогательные значения для постобработки
        self.forces = np.array([F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE], dtype=np.float64)
        self.currents = np.array(
            [i_Na, i_CaL, i_NaCa, i_NaK, i_K1, i_Kr,
             i_Ks, i_K_ATP, i_to, i_rel, i_up, i_leak],
            dtype=np.float64,
        )

        return np.array([
            dd, df2, dfCass, df,
            dCa_SR, dCa_i, dCa_ss, dp_iup,
            dh, dj, dm, dV, dK_i,
            dXr1, dXr2, dXs, dNa_i, dr, ds,
            dv, dw, dN, dA, dl_1, dl_2, dl_3,
            dR, dO, dI, dRI,
            dCaMKt, dCa_nSR, dCa_jSR,
        ])

    # ------------------------------------------------------------------
    # Подсистема: кальциевая динамика
    # ------------------------------------------------------------------

    def _calcium_subsystem(
        self,
        Ca_i, Ca_ss, Ca_jSR, Ca_nSR, Ca_SR,
        d, f, f2, fCass, V, Na_i, CaMKt, p_iup,
        N, A, R, O, I, RI,
    ):
        """Кальциевая динамика: CaMK, SERCA, RyR, SR, Ca_i, Ca_ss."""
        KmCaMK = 0.15
        aCaMK = 0.05
        bCaMK = 0.00068
        CaMKo = 0.05
        KmCaM = 0.0015

        CaMKb = CaMKo * (1.0 - CaMKt) / (1.0 + KmCaM / Ca_ss)
        CaMKa = CaMKb + CaMKt
        dCaMKt = aCaMK * CaMKb * (CaMKb + CaMKt) - bCaMK * CaMKt
        fJupp = 1.0 / (1.0 + KmCaMK / CaMKa)

        Jup_Multiplier = 1.0
        Jupnp = Jup_Multiplier * self.Vmax_up / (1.0 + self.K_up**2.0 / Ca_i**2.0)
        Jupp = Jup_Multiplier * 2.75 * self.Vmax_up / (
            1.0 + (self.K_up - 0.00017) ** 2.0 / Ca_i**2.0
        )
        i_leak = Jup_Multiplier * self.V_leak * (Ca_nSR - Ca_i)
        i_up = (1.0 - fJupp) * Jupnp + fJupp * Jupp

        i_tr = (Ca_nSR - Ca_jSR) / 60.0
        dCa_nSR = i_up - i_leak - i_tr * self.V_jSR / self.V_nSR

        # RyR 4-состоянная модель Shannon
        k_CaSR = self.max_sr - (self.max_sr - self.min_sr) / (
            1.0 + (self.EC / Ca_jSR) ** 2.0
        )
        k_o_SR_Ca = self.k_o_Ca / k_CaSR
        k_i_SR_Ca = self.k_i_Ca * k_CaSR

        dR = (self.k_im * RI - k_i_SR_Ca * R * Ca_ss
              - k_o_SR_Ca * R * Ca_ss**2.0 + self.k_om * O)
        dO = (k_o_SR_Ca * R * Ca_ss**2.0 - self.k_om * O
              - k_i_SR_Ca * O * Ca_ss + self.k_im * I)
        dI = (k_i_SR_Ca * O * Ca_ss - self.k_im * I
              - self.k_om * I + k_o_SR_Ca * RI * Ca_ss**2.0)
        dRI = (self.k_om * I - k_o_SR_Ca * RI * Ca_ss**2.0
               - self.k_im * RI + k_i_SR_Ca * R * Ca_ss)

        # SERCA (Semin p_iup)
        dp_iup = self.k_sm_p * (Ca_i**2.0 * (1 - p_iup) - self.K_lrg_p**2.0 * p_iup)

        i_rel = self.V_rel * O * (Ca_jSR - Ca_ss)
        i_xfer = self.V_xfer * (Ca_ss - Ca_i)

        Ca_jsr_bufsr = 1.0 / (
            1.0 + self.Buf_jsr * self.K_buf_jsr / (Ca_jSR + self.K_buf_jsr) ** 2.0
        )
        dCa_jSR = Ca_jsr_bufsr * (i_tr - i_rel)

        # Устаревшая Ca_SR (dCa_SR = 0 — заглушка при переходе на split SR)
        dCa_SR = 0.0

        # i_NaCa (нужен для dCa_i)
        i_NaCa = self._i_NaCa(V, Na_i, Ca_i)

        # i_b_Ca, i_p_Ca
        E_Ca = 0.5 * self.R * self.T / self.F * log(self.Ca_o / Ca_i)
        i_b_Ca = self.g_bCa * (V - E_Ca)
        i_p_Ca = self.g_pCa * Ca_i / (Ca_i + self.K_pCa)

        # Связывание Ca²⁺ с TnC
        A_off = self.a_off * self.pi_N_A(N, A) * exp(-self.k_A * A)
        dA = self.a_on * (self.A_tot - A) * Ca_i - A_off * A

        Ca_i_bufc = 1.0 / (
            1.0 + self.Buf_c * self.K_buf_c / (Ca_i + self.K_buf_c) ** 2.0
        )
        dCa_i = Ca_i_bufc * (
            (i_leak - i_up) * self.V_nSR / self.V_c
            + i_xfer
            - (i_b_Ca + i_p_Ca - 2.0 * i_NaCa) * self.Cm / (2.0 * self.V_c * self.F)
            - dA
        )

        # i_CaL
        i_CaL = (
            self.g_CaL * d * f * f2 * fCass
            * 4.0 * (V - 15.0) * self.F**2.0 / (self.R * self.T)
            * (0.25 * Ca_ss * exp(2.0 * (V - 15.0) * self.F / (self.R * self.T)) - self.Ca_o)
            / (exp(2.0 * (V - 15.0) * self.F / (self.R * self.T)) - 1.0)
        )

        Ca_ss_bufss = 1.0 / (
            1.0 + self.Buf_ss * self.K_buf_ss / (Ca_ss + self.K_buf_ss) ** 2.0
        )
        dCa_ss = Ca_ss_bufss * (
            -i_CaL * self.Cm / (2.0 * self.V_ss * self.F)
            + i_rel * self.V_jSR / self.V_ss
            - i_xfer * self.V_c / self.V_ss
        )

        return (
            dCaMKt, i_up, i_leak, i_tr, i_rel, i_xfer,
            dCa_nSR, dCa_jSR, dCa_SR, dCa_i, dCa_ss,
            i_CaL, dp_iup, dR, dO, dI, dRI,
        )

    # ------------------------------------------------------------------
    # Подсистема: электрофизиология
    # ------------------------------------------------------------------

    def _electrophysiology_subsystem(
        self,
        V, d, f2, fCass, f, h, j, m, Xr1, Xr2, Xs, r, s,
        K_i, Na_i, Ca_i, Ca_ss,
    ):
        """Ионные токи и ворота TNNP."""
        E_Na = self.R * self.T / self.F * log(self.Na_o / Na_i)
        E_K = self.R * self.T / self.F * log(self.K_o / K_i)

        # --- Ворота L-type Ca²⁺ ---
        d_inf = 1.0 / (1.0 + exp((-8.0 - V) / 7.5))
        alpha_d = 1.4 / (1.0 + exp((-35.0 - V) / 13.0)) + 0.25
        beta_d = 1.4 / (1.0 + exp((V + 5.0) / 5.0))
        gamma_d = 1.0 / (1.0 + exp((50.0 - V) / 20.0))
        dd = (d_inf - d) / (alpha_d * beta_d + gamma_d)

        f2_inf = 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33
        tau_f2 = (
            562.0 * exp(-((V + 27.0) ** 2.0) / 240.0)
            + 31.0 / (1.0 + exp((25.0 - V) / 10.0))
            + 80.0 / (1.0 + exp((V + 30.0) / 10.0))
        )
        df2 = (f2_inf - f2) / tau_f2

        fCass_inf = 0.6 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 0.4
        tau_fCass = 80.0 / (1.0 + (Ca_ss / 0.05) ** 2.0) + 2.0
        dfCass = (fCass_inf - fCass) / tau_fCass

        f_inf = 1.0 / (1.0 + exp((V + 20.0) / 7.0))
        tau_f = (
            1102.5 * exp(-((V + 27.0) ** 2.0) / 225.0)
            + 200.0 / (1.0 + exp((13.0 - V) / 10.0))
            + 180.0 / (1.0 + exp((V + 30.0) / 10.0))
            + 20.0
        )
        df = (f_inf - f) / tau_f

        # --- Ворота Na⁺ ---
        h_inf = 1.0 / (1.0 + exp((V + 71.55) / 7.43)) ** 2.0
        alpha_h = 0.057 * exp(-(V + 80.0) / 6.8) if V < -40.0 else 0.0
        beta_h = (
            2.7 * exp(0.079 * V) + 310000.0 * exp(0.3485 * V)
            if V < -40.0
            else 0.77 / (0.13 * (1.0 + exp((V + 10.66) / -11.1)))
        )
        dh = (h_inf - h) / (1.0 / (alpha_h + beta_h))

        j_inf = h_inf
        alpha_j = (
            ((-25428.0 * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V))
             * (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23))))
            if V < -40.0 else 0.0
        )
        beta_j = (
            0.02424 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14)))
            if V < -40.0
            else 0.6 * exp(0.057 * V) / (1.0 + exp(-0.1 * (V + 32.0)))
        )
        dj = (j_inf - j) / (1.0 / (alpha_j + beta_j))

        m_inf = 1.0 / (1.0 + exp((-56.86 - V) / 9.03)) ** 2.0
        alpha_m = 1.0 / (1.0 + exp((-60.0 - V) / 5.0))
        beta_m = 0.1 / (1.0 + exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + exp((V - 50.0) / 200.0))
        dm = (m_inf - m) / (alpha_m * beta_m)

        # --- i_Na ---
        i_Na = self.g_Na * (m**3.0) * h * j * (V - E_Na)

        # --- i_K1 ---
        alpha_K1 = 0.1 / (1.0 + exp(0.06 * (V - E_K - 200.0)))
        beta_K1 = (
            (3.0 * exp(0.0002 * (V - E_K + 100.0)) + exp(0.1 * (V - E_K - 10.0)))
            / (1.0 + exp(-0.5 * (V - E_K)))
        )
        xK1_inf = alpha_K1 / (alpha_K1 + beta_K1)
        i_K1 = self.g_K1 * xK1_inf * sqrt(self.K_o / 5.4) * (V - E_K)

        # --- i_to ---
        i_to = self.g_to * r * s * (V - E_K)

        # --- i_Kr ---
        i_Kr = self.g_Kr * sqrt(self.K_o / 5.4) * Xr1 * Xr2 * (V - E_K)

        # --- i_Ks ---
        E_Ks = self.R * self.T / self.F * log(
            (self.K_o + self.P_kna * self.Na_o) / (K_i + self.P_kna * Na_i)
        )
        i_Ks = self.g_Ks * (Xs**2.0) * (V - E_Ks)

        # --- i_K_ATP ---
        PATP = 1.0 / (1.0 + (self.ATPi / self.KmATP) ** 2.2)
        i_K_ATP = self.g_K_ATP * PATP * ((self.K_o / 5.4) ** 0.24) * (V - E_K)

        # --- i_NaK ---
        i_NaK = (
            self.P_NaK * self.K_o * Na_i
            / (
                (self.K_o + self.K_mk)
                * (Na_i + self.K_mNa)
                * (1.0
                   + 0.1245 * exp(-0.1 * V * self.F / (self.R * self.T))
                   + 0.0353 * exp(-V * self.F / (self.R * self.T)))
            )
        )

        # --- i_NaCa ---
        i_NaCa = self._i_NaCa(V, Na_i, Ca_i)

        # --- Фоновые токи ---
        i_b_Na = self.g_bna * (V - E_Na)
        i_p_K = self.g_pK * (V - E_K) / (1.0 + exp((25.0 - V) / 5.98))
        E_Ca = 0.5 * self.R * self.T / self.F * log(self.Ca_o / Ca_i)
        i_b_Ca = self.g_bCa * (V - E_Ca)
        i_p_Ca = self.g_pCa * Ca_i / (Ca_i + self.K_pCa)

        # --- Ворота K⁺ ---
        xr1_inf = 1.0 / (1.0 + exp((-26.0 - V) / 7.0))
        alpha_xr1 = 450.0 / (1.0 + exp((-45.0 - V) / 10.0))
        beta_xr1 = 6.0 / (1.0 + exp((V + 30.0) / 11.5))
        dXr1 = (xr1_inf - Xr1) / (alpha_xr1 * beta_xr1)

        xr2_inf = 1.0 / (1.0 + exp((V + 88.0) / 24.0))
        alpha_xr2 = 3.0 / (1.0 + exp((-60.0 - V) / 20.0))
        beta_xr2 = 1.12 / (1.0 + exp((V - 60.0) / 20.0))
        dXr2 = (xr2_inf - Xr2) / (alpha_xr2 * beta_xr2)

        xs_inf = 1.0 / (1.0 + exp((-5.0 - V) / 14.0))
        alpha_xs = 1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0))
        beta_xs = 1.0 / (1.0 + exp((V - 35.0) / 15.0))
        dXs = (xs_inf - Xs) / (alpha_xs * beta_xs + 80.0)

        r_inf = 1.0 / (1.0 + exp((20.0 - V) / 6.0))
        tau_r = 9.5 * exp(-((V + 40.0) ** 2.0) / 1800.0) + 0.8
        dr = (r_inf - r) / tau_r

        s_inf = 1.0 / (1.0 + exp((V + 20.0) / 5.0))
        tau_s = (
            85.0 * exp(-((V + 45.0) ** 2.0) / 320.0)
            + 5.0 / (1.0 + exp((V - 20.0) / 5.0))
            + 3.0
        )
        ds = (s_inf - s) / tau_s

        return (
            i_Na, i_NaCa, i_NaK, i_K1, i_to, i_Kr,
            i_Ks, i_K_ATP, i_b_Na, i_b_Ca, i_p_Ca, i_p_K,
            dd, df2, dfCass, df, dh, dj, dm,
            dXr1, dXr2, dXs, dr, ds,
        )

    # ------------------------------------------------------------------
    # Подсистема: мембранный потенциал и ионные концентрации
    # ------------------------------------------------------------------

    def _membrane_subsystem(
        self, time, V, K_i, Na_i,
        i_K1, i_to, i_Kr, i_Ks, i_CaL, i_NaK, i_Na,
        i_b_Na, i_NaCa, i_b_Ca, i_p_K, i_p_Ca, i_K_ATP,
    ):
        """dV, dK_i, dNa_i."""
        t_mod = time - floor(time / self.stim_period) * self.stim_period
        i_Stim = (
            -self.stim_amplitude
            if self.stim_start <= t_mod <= self.stim_start + self.stim_duration
            else 0.0
        )

        dV = -(
            i_K1 + i_to + i_Kr + i_Ks + i_CaL + i_NaK
            + i_Na + i_b_Na + i_NaCa + i_b_Ca
            + i_p_K + i_p_Ca + i_K_ATP + i_Stim
        )

        dK_i = (
            -(i_K1 + i_to + i_Kr + i_Ks + i_p_K + i_Stim - 2.0 * i_NaK + i_K_ATP)
            / (self.V_c * self.F) * self.Cm
        )

        dNa_i = (
            -(i_Na + i_b_Na + 3.0 * i_NaK + 3.0 * i_NaCa)
            / (self.V_c * self.F) * self.Cm
        )

        return dV, dK_i, dNa_i

    # ------------------------------------------------------------------
    # Подсистема: механика саркомера
    # ------------------------------------------------------------------

    def _mechanics_subsystem(self, v, w, N, A, l_1, l_2, l_3, Ca_i):
        """Механика: поперечные мостики, длины элементов, силы.

        Returns:
            Кортеж (dN, dA, dv, dw, dl_1, dl_2, dl_3, forces),
            где forces = (F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE).
        """
        machine_zero = 1e-15

        K_chi = (
            self.k_p_v(v) * self.M(A) * self.n_1(l_1) * self.L_oz(l_1) * (1.0 - N)
            - self.k_m_v(v) * N
        )
        dN = K_chi

        A_off = self.a_off * self.pi_N_A(N, A) * exp(-self.k_A * A)
        dA = self.a_on * (self.A_tot - A) * Ca_i - A_off * A

        F_muscle = self.beta_3 * (exp(self.alpha_3 * l_3) - 1.0)
        l = l_2 + l_3

        isotonic_mode = (
            self.F_afterload > machine_zero
            and F_muscle >= self.F_afterload
            and l <= self.l0 * (1.0 + 1.0e-4)
            and F_muscle > self.r0
        )

        # Вязкоупругие коэффициенты
        alpha_p = self.alpha_vp_l if v <= 0.0 else self.alpha_vp_s
        k_P_vis = (
            self.beta_vp_l * exp(self.alpha_vp_l * l_1)
            if v <= 0.0
            else self.beta_vp_s * exp(self.alpha_vp_s * l_1)
        )
        alpha_s = self.alpha_vs_l if w <= v else self.alpha_vs_s
        k_S_vis = (
            self.beta_vs_l * exp(self.alpha_vs_l * (l_2 - l_1))
            if w <= v
            else self.beta_vs_s * exp(self.alpha_vs_s * (l_2 - l_1))
        )

        # dv (скорость CE)
        common_num_v = (
            self.llambda * K_chi * self.p_v(v)
            + alpha_p * k_P_vis * v**2.0
        )
        if isotonic_mode:
            phi_chi = -(
                common_num_v + self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2) * w
            ) / (self.llambda * N * self.p_prime_v(v) + k_P_vis)
        else:
            phi_chi = -(
                common_num_v
                + (self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2)
                   + self.alpha_3 * self.beta_3 * exp(self.alpha_3 * l_3)) * w
            ) / (self.llambda * N * self.p_prime_v(v) + k_P_vis)
        dv = phi_chi

        # dw (скорость PE)
        se_term = self.alpha_1 * self.beta_1 * exp(self.alpha_1 * (l_2 - l_1)) * (w - v)
        pe_term = self.alpha_2 * self.beta_2 * exp(self.alpha_2 * l_2) * w
        xse_term = self.alpha_3 * self.beta_3 * exp(self.alpha_3 * l_3) * w
        if isotonic_mode:
            dw = (phi_chi - alpha_s * (w - v) ** 2.0
                  - (se_term + pe_term) / k_S_vis)
        else:
            dw = (phi_chi - alpha_s * (w - v) ** 2.0
                  - (se_term + pe_term + xse_term) / k_S_vis)

        dl_1 = v
        dl_2 = w
        dl_3 = 0.0 if isotonic_mode else -w

        F_CE = self.llambda * self.p_v(v) * N
        F_SE = self.beta_1 * (exp(self.alpha_1 * (l_2 - l_1)) - 1.0)
        F_PE = self.beta_2 * (exp(self.alpha_2 * l_2) - 1.0)
        F_VS1 = k_P_vis * v
        F_VS2 = k_S_vis * (w - v)
        F_XSE = self.beta_3 * (exp(self.alpha_3 * l_3) - 1.0)

        return dN, dA, dv, dw, dl_1, dl_2, dl_3, (F_CE, F_SE, F_PE, F_VS1, F_VS2, F_XSE)

    # ------------------------------------------------------------------
    # Вспомогательный ток i_NaCa
    # ------------------------------------------------------------------

    def _i_NaCa(self, V, Na_i, Ca_i):
        """Na⁺/Ca²⁺ обменник."""
        num = (
            exp(self.gamma * V * self.F / (self.R * self.T)) * (Na_i**3.0) * self.Ca_o
            - exp((self.gamma - 1.0) * V * self.F / (self.R * self.T))
            * (self.Na_o**3.0) * Ca_i * self.alpha
        )
        den = (
            (self.Km_Nai**3.0 + self.Na_o**3.0)
            * (self.Km_Ca + self.Ca_o)
            * (1.0 + self.K_sat * exp(
                (self.gamma - 1.0) * V * self.F / (self.R * self.T)
            ))
        )
        return self.K_NaCa * num / den

    # ------------------------------------------------------------------
    # Вспомогательные функции механики
    # ------------------------------------------------------------------

    def k_p_v(self, v: float) -> float:
        """Скорость прикрепления поперечных мостиков."""
        return self.chi(v) * self.chi_0 * self.q_v(v) * self.m_0 * self.G_star(v / self.v_max)

    def k_m_v(self, v: float) -> float:
        """Скорость отсоединения поперечных мостиков."""
        return (
            self.chi_0 * self.q_v(v)
            * (1.0 - self.chi(v) * self.m_0 * self.G_star(v / self.v_max))
        )

    def M(self, A: float) -> float:
        """Кооперативность активации миозина через TnC."""
        ratio = (A / self.A_tot) ** self.mu
        k_ratio = self.k_mu ** self.mu
        return ratio * (1.0 + k_ratio) / (ratio + k_ratio)

    def n_1(self, l_1: float) -> float:
        """Перекрытие нитей в зависимости от длины PE."""
        w1 = (self.g_1 * l_1 + self.g_2) * (
            self.n1_A
            + (self.n1_K - self.n1_A)
            / (self.n1_C + self.n1_Q * exp(-self.n1_B * l_1)) ** (1.0 / self.n1_nu)
        )
        return max(0.0, min(1.0, w1))

    def L_oz(self, l_1: float) -> float:
        """Фактор перекрытия актин-миозин."""
        if l_1 <= self.s055:
            return (l_1 + self.s_0) / (self.s046 + self.s_0)
        return (self.s_0 + self.s055) / (self.s046 + self.s_0)

    def chi(self, v: float) -> float:
        """Зависимость скорости прикрепления от скорости CE."""
        if v <= 0.0:
            return self.chi_1 + self.chi_2 * v / self.v_max
        return self.chi_1

    def q_v(self, v: float) -> float:
        """Нормализованная сила поперечных мостиков от скорости."""
        if v <= 0.0:
            return self.q_1 - self.q_2 * v / self.v_max
        elif v <= self.x_st * self.v_max:
            return (self.q_4 - self.q_3) * v / (self.x_st * self.v_max) + self.q_3
        else:
            return self.q_4 / (
                1.0 + self.beta_Q * (v / self.v_max - self.x_st)
            ) ** self.alpha_Q

    def G_star(self, v: float) -> float:
        """Нормализованная сила CE."""
        den = (0.4 * self.a + 1.0) * v / self.a + 1.0
        if v <= 0:
            return 1.0 + 0.6 * v
        elif v <= self.v_1:
            return self.P_star(v) / den
        else:
            return (
                self.P_star(v)
                * exp(-self.alpha_G * (v - self.v_1) ** self.alpha_P)
                / den
            )

    def P_star(self, v: float) -> float:
        """Зависимость P от нормированной скорости CE."""
        gamma = (
            self.a * self.d_h * (self.v_1**2.0)
            / (3.0 * self.a * self.d_h - (self.a + 1.0) * self.v_1)
        )
        return (
            1.0 + self.d_h
            - (self.d_h**2.0) * self.a
            / ((self.a + 1.0) * v + self.d_h * self.a
               + self.a * self.d_h * (v**2.0) / gamma)
        )

    def pi_N_A(self, N: float, A: float) -> float:
        """Кооперативный фактор для отсоединения Ca²⁺ от TnC."""
        if A <= 0.0:
            return 1.0
        N_A = self.A_tot * self.s_c * N / A
        if N_A <= 0.0:
            return 1.0
        elif N_A <= 1.0:
            return self.pi_min**N_A
        else:
            return self.pi_min

    def p_v(self, v: float) -> float:
        """Нормализованная сила CE (Hill-подобная функция)."""
        if v <= -self.v_max:
            return 0.0
        elif v <= 0.0:
            return (
                self.a * (1.0 + v / self.v_max)
                / ((self.a - v / self.v_max) * (1.0 + 0.6 * v / self.v_max))
            )
        elif v <= self.v_1 * self.v_max:
            return (0.4 * self.a + 1.0) * v / (self.a * self.v_max) + 1.0
        else:
            return (
                ((0.4 * self.a + 1.0) * v / (self.a * self.v_max) + 1.0)
                * exp(self.alpha_G * (v / self.v_max - self.v_1) ** self.alpha_P)
            )

    def p_prime_v(self, v: float) -> float:
        """Производная p_v по v."""
        if v <= -self.v_max:
            return self.a * (0.4 + 0.4 * self.a) / (
                self.v_max * ((self.a + 1.0) * 0.4) ** 2.0
            )
        elif v <= 0.0:
            return (
                self.a
                * (1.0 + 0.4 * self.a + 1.2 * v / self.v_max + 0.6 * (v / self.v_max) ** 2.0)
                / (self.v_max * (
                    (self.a - v / self.v_max) * (1.0 + 0.6 * v / self.v_max)
                ) ** 2.0)
            )
        elif v <= self.v_1 * self.v_max:
            return (0.4 * self.a + 1.0) / (self.a * self.v_max)
        else:
            return (
                exp(self.alpha_G * (v / self.v_max - self.v_1) ** self.alpha_P)
                * (
                    (0.4 * self.a + 1.0) / self.a
                    + self.alpha_G * self.alpha_P
                    * (1.0 + (0.4 * self.a + 1.0) * v / (self.a * self.v_max))
                    * (v / self.v_max - self.v_1) ** (self.alpha_P - 1.0)
                )
                / self.v_max
            )

    def N0(self, l_0: float) -> float:
        """Равновесная доля поперечных мостиков при длине l_0."""
        return (self.r0 - self.beta_2 * (exp(self.alpha_2 * l_0) - 1.0)) / self.llambda

    def fi(self, l_1: float) -> float:
        """Целевая функция для бисекции (равновесие механики)."""
        return (
            self.k_p_v(0) * self.M(6.31929074e-04) * self.n_1(l_1) * self.L_oz(l_1)
            * (1.0 - self.N0(l_1))
            - self.k_m_v(0) * self.N0(l_1)
        )

    def L(self, l_0: float) -> float:
        """Равновесная длина PE при заданной l_0."""
        return l_0 + (
            log(self.beta_1)
            - log(self.r0 + self.beta_1 - self.beta_2 * (exp(self.alpha_2 * l_0) - 1.0))
        ) / self.alpha_1

    def delenie(self) -> float:
        """Найти равновесную длину l_2 методом бисекции."""
        l_max = log((self.r0 + self.beta_2) / self.beta_2) / self.alpha_2
        a = 0.9 * l_max
        b = l_max

        if self.fi(a) == 0.0:
            return a
        if self.fi(b) == 0.0:
            return b

        x = a + (b - a) / 2.0
        while abs(self.fi(x)) >= 1e-7:
            x = a + (b - a) / 2.0
            if self.fi(x) < 0:
                a = x
            else:
                b = x
        return x

    def calculate_init_conditions(self) -> np.ndarray:
        """Вычислить начальные условия с механическим равновесием.

        Обновляет атрибуты ``v_init``, ``w_init``, ``N_init``,
        ``l_1_init``, ``l_2_init``, ``l_3_init``, ``l0``,
        ``F_afterload``.

        Returns:
            Массив начальных условий длиной 33.
        """
        l_2 = self.delenie()
        l_1 = self.L(l_2)
        N = self.N0(l_2)
        l_3 = log((self.r0 + self.beta_3) / self.beta_3) / self.alpha_3

        self.v_init = 0.0
        self.w_init = 0.0
        self.N_init = N
        self.l_1_init = l_1
        self.l_2_init = l_2
        self.l_3_init = l_3
        self.l0 = l_2 + l_3
        self.F_afterload = 0.0

        return np.array([
            self.d_init, self.f2_init, self.fCass_init, self.f_init,
            self.Ca_SR_init, self.Ca_i_init, self.Ca_ss_init, self.p_iup,
            self.h_init, self.j_init, self.m_init, self.V_init,
            self.K_i_init, self.Xr1_init, self.Xr2_init, self.Xs_init,
            self.Na_i_init, self.r_init, self.s_init,
            self.v_init, self.w_init, self.N_init, self.A_init,
            self.l_1_init, self.l_2_init, self.l_3_init,
            self.R_init, self.O_init, self.I_init, self.RI_init,
            self.CaMKt_init, self.Ca_nSR_init, self.Ca_jSR_init,
        ])
