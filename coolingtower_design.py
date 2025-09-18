# streamlit_app.py
# Cooling Tower Thermal Design — Merkel‑lite first‑pass sizing
# Author: <you>
# License: MIT

import math
import json
from io import BytesIO
from dataclasses import dataclass

import streamlit as st

# -------------------------------
# Utilities
# -------------------------------
P_ATM = 101325.0  # Pa
CP_W = 4180.0     # J/(kg·K) for liquid water
R_DA = 287.055    # J/(kg·K) for dry air
R_WV = 461.495    # J/(kg·K) for water vapor

# Tetens saturation vapor pressure (Pa), good ~0–50°C
# Psat(T) ≈ 610.78 * exp(17.2694*T / (T+237.3)) where T is in °C
# Use for psychrometrics; for wet-bulb solve we use standard formulas.
def psat_Pa(T_C: float) -> float:
    return 610.78 * math.exp(17.2694 * T_C / (T_C + 237.3))

# Humidity ratio from DB and WB (°C), pressure Pa
# Using iterative approach from ASHRAE‑style psychrometrics (approximate)
def humidity_ratio_from_DB_WB(Tdb_C: float, Twb_C: float, P=P_ATM) -> float:
    # psychrometric constant gamma ≈ (Cp_air * P) / (0.622 * h_fg)
    # We'll estimate Cp_air ≈ 1006 J/kgK, h_fg at Twb
    Cp_air = 1006.0
    h_fg = 2501000.0 - 2369.0 * Twb_C  # J/kg, simple linear approx
    gamma = (Cp_air * P) / (0.622 * h_fg)
    Pw_s_wb = psat_Pa(Twb_C)
    # Psychrometric relation: Pw = Pw* - gamma*(Tdb - Twb)
    Pw = Pw_s_wb - gamma * (Tdb_C - Twb_C)
    Pw = max(1.0, min(Pw, 0.99 * P))
    W = 0.622 * Pw / (P - Pw)
    return max(W, 1e-6)

# Moist air enthalpy (kJ/kg dry air)
def h_moist_air_kJ_per_kg_da(T_C: float, W: float) -> float:
    return 1.006 * T_C + W * (2501.0 + 1.86 * T_C)

# Moist air density from T (°C), W (kg/kg), P (Pa)
def rho_moist_air(T_C: float, W: float, P=P_ATM) -> float:
    T_K = T_C + 273.15
    # Ideal gas mix: rho = P / (R_da * T) * (1 + W/0.622) / (1 + W)
    return (P / (R_DA * T_K)) * (1 + W / 0.622) / (1 + W)

# Saturated air enthalpy at water temperature Tw (°C) at pressure P
# (used for Merkel driving potential i*_s(Tw))
def h_star_kJ_per_kg_da(Tw_C: float, P=P_ATM) -> float:
    Pw_star = psat_Pa(Tw_C)
    W_star = 0.622 * Pw_star / (P - Pw_star)
    return h_moist_air_kJ_per_kg_da(Tw_C, W_star)

# -------------------------------
# Fill database (editable)
# -------------------------------
# Note: Default values are engineering placeholders. Replace with vendor data.
DEFAULT_FILLS = {
    "Brentwood CF1900 (crossflow film)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 1.6,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 5.0,
        "rec_water_loading_m3_h_m2_max": 18.0,
        "free_area_frac": 0.90,
        "depth_m_default": 0.6,
        # Pressure drop model: dP = k0 * (v/vr)^2 * depth_m   [Pa]
        "dp_k0_Pa_per_m_at_vr": 110.0,
        "dp_vr_m_s": 2.2,
    },
    "Brentwood OF21MA (counterflow film)": {
        "vendor": "Brentwood",
        "geometry": "film",
        "flow": "counterflow",
        "rec_air_velocity_m_s_min": 2.0,
        "rec_air_velocity_m_s_max": 3.5,
        "rec_water_loading_m3_h_m2_min": 6.0,
        "rec_water_loading_m3_h_m2_max": 20.0,
        "free_area_frac": 0.92,
        "depth_m_default": 0.9,
        "dp_k0_Pa_per_m_at_vr": 150.0,
        "dp_vr_m_s": 2.5,
    },
    "Brentwood XF75 (crossflow splash/film hybrid)": {
        "vendor": "Brentwood",
        "geometry": "hybrid",
        "flow": "crossflow",
        "rec_air_velocity_m_s_min": 1.8,
        "rec_air_velocity_m_s_max": 3.0,
        "rec_water_loading_m3_h_m2_min": 4.0,
        "rec_water_loading_m3_h_m2_max": 14.0,
        "free_area_frac": 0.88,
        "depth_m_default": 0.6,
        "dp_k0_Pa_per_m_at_vr": 95.0,
        "dp_vr_m_s": 2.2,
    },
}

# -------------------------------
# Merkel‑lite core: estimate air flow from heat load and enthalpy rise
# -------------------------------

def suggest_air_flow(Q_kW: float, T_w_out_C: float, T_db_in_C: float, T_wb_in_C: float):
    """
    Suggest dry‑air mass flow [kg/s] and volumetric flow [m^3/s] from heat load and moist‑air enthalpy rise.
    Assumptions:
      • Outlet air is near saturation at temperature ~ T_w_out + 1.5°C (empirical rule of thumb for pack fills)
      • Inlet air state from DB/WB
    """
    W_in = humidity_ratio_from_DB_WB(T_db_in_C, T_wb_in_C)
    h_in = h_moist_air_kJ_per_kg_da(T_db_in_C, W_in)  # kJ/kg_da

    T_exh = T_w_out_C + 1.5  # °C, tweak in Advanced settings if needed
    # Assume 95% RH at exhaust if T_exh < DB_in, otherwise cap at saturation
    Pw_star = psat_Pa(T_exh)
    RH_exh = 0.95
    Pw_exh = min(RH_exh * Pw_star, 0.99 * P_ATM)
    W_exh = 0.622 * Pw_exh / (P_ATM - Pw_exh)
    h_out = h_moist_air_kJ_per_kg_da(T_exh, W_exh)

    delta_h = max(h_out - h_in, 0.5)  # kJ/kg_da
    G_da = (Q_kW) / delta_h  # kg_dry_air / s  (since 1 kW = 1 kJ/s)

    # Density at mean state (rough)
    T_mean = 0.5 * (T_db_in_C + T_exh)
    W_mean = 0.5 * (W_in + W_exh)
    rho = rho_moist_air(T_mean, W_mean)
    # Convert dry‑air mass flow to total moist‑air mass flow
    m_dot_moist = G_da * (1.0 + W_mean)
    V_dot = m_dot_moist / rho  # m^3/s
    return {
        "W_in": W_in,
        "W_out": W_exh,
        "h_in_kJkg": h_in,
        "h_out_kJkg": h_out,
        "G_da_kg_s": G_da,
        "rho_mean": rho,
        "V_dot_m3_s": V_dot,
        "T_exh_C": T_exh,
    }


def pressure_drop_fill(v_face: float, depth_m: float, dp_k0: float, v_ref: float) -> float:
    """Quadratic scaling with velocity; linear with depth."""
    return dp_k0 * (v_face / v_ref) ** 2 * (depth_m / 1.0)


# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(page_title="Cooling Tower Sizer — Merkel‑lite", layout="wide")
st.title("🧊 Cooling Tower Thermal Sizer — Merkel‑lite (first‑pass)")
st.caption(
    "Engineering estimator for air flow, fill area, velocities, and pressure drop. Replace defaults with vendor data for production use."
)

with st.sidebar:
    st.header("Project & Options")
    proj_name = st.text_input("Project name", "Demo Tower")
    flow_type = st.radio("Tower flow arrangement", ["counterflow", "crossflow"], index=1)

    # Editable fill library
    st.subheader("Fill library (JSON, editable)")
    fills_json = st.text_area(
        "Paste/adjust fill database (per‑model coefficients)",
        value=json.dumps(DEFAULT_FILLS, indent=2),
        height=300,
    )
    try:
        fills_db = json.loads(fills_json)
    except Exception as e:
        st.error(f"Fill JSON parse error: {e}")
        fills_db = DEFAULT_FILLS

    # Filter fills by flow type
    fill_names = [k for k, v in fills_db.items() if v.get("flow", "")==flow_type]
    if not fill_names:
        fill_names = list(fills_db.keys())
    fill_name = st.selectbox("Select fill model", fill_names)
    fill = fills_db[fill_name]

    depth_m = st.number_input("Fill depth (m)", 0.2, 2.0, float(fill.get("depth_m_default", 0.6)), 0.05)
    v_face_target = st.number_input(
        "Target air face velocity through fill (m/s)",
        0.5,
        5.0,
        float(0.5 * (fill.get("rec_air_velocity_m_s_min", 1.5) + fill.get("rec_air_velocity_m_s_max", 3.0))),
        0.05,
    )
    free_area_frac = st.slider("Fill free‑area fraction", 0.6, 0.98, float(fill.get("free_area_frac", 0.9)), 0.01)
    dp_k0 = float(fill.get("dp_k0_Pa_per_m_at_vr", 110.0))
    v_ref = float(fill.get("dp_vr_m_s", 2.2))

st.subheader("Water & Air Inputs")
col1, col2, col3, col4 = st.columns(4)
with col1:
    Qw_L_min = st.number_input("Water flow (L/min)", min_value=10.0, max_value=100000.0, value=6000.0, step=10.0)
    m_dot_w = Qw_L_min / 60.0 * 1e-3 * 1000.0  # kg/s (ρ≈1000)
    st.write(f"**Water mass flow:** {m_dot_w:.2f} kg/s")
with col2:
    Tin_C = st.number_input("Hot water in (°C)", min_value=10.0, max_value=80.0, value=37.0, step=0.1)
with col3:
    Tdb_C = st.number_input("Ambient dry‑bulb DB (°C)", min_value=-10.0, max_value=55.0, value=38.0, step=0.1)
with col4:
    Twb_C = st.number_input("Ambient wet‑bulb WB (°C)", min_value=-10.0, max_value=35.0, value=28.0, step=0.1)

st.markdown("---")

mode = st.radio("Sizing mode", ["Specify outlet temperature", "Specify heat load (kW)"])

if mode == "Specify outlet temperature":
    Tout_C = st.number_input("Required cold water out (°C)", min_value=5.0, max_value=Tin_C-0.1, value=max(Twb_C+4.0, 25.0), step=0.1)
    Q_kW = (m_dot_w * CP_W * (Tin_C - Tout_C)) / 1000.0
    st.write(f"**Heat rejected (from water):** {Q_kW:.1f} kW")
else:
    Q_kW = st.number_input("Heat to reject (kW)", min_value=5.0, max_value=20000.0, value=3500.0, step=5.0)
    # Compute achievable outlet temp (first pass, using enthalpy‑based air flow suggestion + approach rule)
    # We will iterate on Tout using a small solver to match Q.
    def find_Tout(Q_kW):
        # Bracket between WB+2 and Tin-0.1
        lo = min(Twb_C + 1.0, Tin_C - 0.1)
        hi = Tin_C - 0.1
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            Q_mid = (m_dot_w * CP_W * (Tin_C - mid)) / 1000.0
            if Q_mid > Q_kW:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    Tout_C = find_Tout(Q_kW)
    st.write(f"**Estimated cold water out:** {Tout_C:.2f} °C (range‑only)")

# Suggest air flow from heat and enthalpy rise
sugg = suggest_air_flow(Q_kW, Tout_C, Tdb_C, Twb_C)
Vdot_suggest = sugg["V_dot_m3_s"]

st.markdown("### Fan Airflow")
colA, colB = st.columns(2)
with colA:
    Vdot_user = st.number_input("Fan volumetric flow (m³/s)", min_value=0.1, max_value=500.0, value=float(Vdot_suggest), step=0.1, format="%.3f")
with colB:
    st.metric("Suggested airflow (m³/s)", f"{Vdot_suggest:.2f}")

# Fill plan area from chosen face velocity and free area
A_fill = Vdot_user / max(v_face_target * free_area_frac, 1e-6)

# Superficial velocities
v_air_face = Vdot_user / max(A_fill * free_area_frac, 1e-9)  # should ≈ v_face_target
v_water = (Qw_L_min / 60.0 / 1000.0) / max(A_fill, 1e-9)  # m/s

# Pressure drop across fill
DP_fill = pressure_drop_fill(v_air_face, depth_m, dp_k0, v_ref)

# Check water loading vs recommended (m^3/h·m^2)
water_loading_m3_h_m2 = (Qw_L_min / 1000.0) * 60.0 / max(A_fill, 1e-9)
rec_w_min = fill.get("rec_water_loading_m3_h_m2_min", 4.0)
rec_w_max = fill.get("rec_water_loading_m3_h_m2_max", 18.0)

# Compute achieved kW with selected fan flow (using actual enthalpy rise at fixed exhaust guess)
sugg_user = suggest_air_flow(Q_kW=Q_kW, T_w_out_C=Tout_C, T_db_in_C=Tdb_C, T_wb_in_C=Twb_C)
# If user airflow differs, scale dry‑air flow proportionally to Vdot
G_sugg = sugg_user["G_da_kg_s"]
V_sugg = sugg_user["V_dot_m3_s"]
G_user = G_sugg * (Vdot_user / max(V_sugg, 1e-9))
# Achievable heat ≈ G_user * Δh
delta_h = sugg_user["h_out_kJkg"] - sugg_user["h_in_kJkg"]
Q_ach_kW = G_user * max(delta_h, 0.1)
# Corresponding achievable cold water temp (from range only)
Tout_ach_C = Tin_C - (Q_ach_kW * 1000.0) / (m_dot_w * CP_W)

# -------------------------------
# Results
# -------------------------------

st.markdown("## 📊 Results")
colR1, colR2, colR3, colR4 = st.columns(4)
with colR1:
    st.metric("Fill plan area (m²)", f"{A_fill:.2f}")
    st.metric("Water superficial vel. (m/s)", f"{v_water:.3f}")
with colR2:
    st.metric("Air face vel. (m/s)", f"{v_air_face:.2f}")
    st.metric("Water loading (m³/h·m²)", f"{water_loading_m3_h_m2:.1f}")
with colR3:
    st.metric("Fill dP (Pa)", f"{DP_fill:.0f}")
    st.metric("Fan flow (m³/s)", f"{Vdot_user:.2f}")
with colR4:
    st.metric("Achievable kW (at current fan)", f"{Q_ach_kW:.0f}")
    st.metric("Achievable cold‑out (°C)", f"{Tout_ach_C:.2f}")

# Warnings / checks
warns = []
if v_air_face < fill.get("rec_air_velocity_m_s_min", 0.0) or v_air_face > fill.get("rec_air_velocity_m_s_max", 99):
    warns.append("Air face velocity outside recommended range for selected fill.")
if water_loading_m3_h_m2 < rec_w_min or water_loading_m3_h_m2 > rec_w_max:
    warns.append("Water loading (m³/h·m²) outside recommended range for selected fill.")
if Tout_ach_C < Twb_C + 1.0:
    warns.append("Predicted cold water temperature approaches wet‑bulb; check assumptions and Merkel characteristic.")

if warns:
    st.warning("\n".join([f"• {w}" for w in warns]))

# -------------------------------
# Intermediate Calculations
# -------------------------------

with st.expander("Show intermediate calculations"):
    st.write("**Psychrometrics (inlet air):**")
    W_in = sugg["W_in"]
    h_in = sugg["h_in_kJkg"]
    rho_mean = sugg["rho_mean"]
    st.write({
        "humidity_ratio_in (kg/kg)": round(W_in, 5),
        "h_in (kJ/kg_da)": round(h_in, 2),
        "rho_mean (kg/m³)": round(rho_mean, 3),
        "assumed_exhaust_T (°C)": round(sugg["T_exh_C"], 2),
        "assumed_W_out (kg/kg)": round(sugg["W_out"], 5),
        "h_out (kJ/kg_da)": round(sugg["h_out_kJkg"], 2),
        "Δh (kJ/kg_da)": round(delta_h, 2),
        "G_dry_air_suggest (kg/s)": round(G_sugg, 3),
        "Vdot_suggest (m³/s)": round(V_sugg, 3),
    })
    st.write("**Fill / hydraulics:**")
    st.write({
        "free_area_frac": free_area_frac,
        "v_face_target (m/s)": v_face_target,
        "v_air_face (m/s)": round(v_air_face, 3),
        "A_fill (m²)": round(A_fill, 3),
        "depth_m": depth_m,
        "dp_coeff_k0 (Pa/m @ v_ref)": dp_k0,
        "v_ref (m/s)": v_ref,
        "dP_fill (Pa)": round(DP_fill, 1),
    })

# -------------------------------
# Fan Modelling (Axial) — vendor curves + affinity laws
# -------------------------------

# Dimensionless fan laws helper functions

def fan_p_rel(q_rel, p0, q_rel_max):
    """Pressure coefficient curve: simple shutoff parabola placeholder.
    p_rel = p0 * (1 - (q/qmax)^2)  for 0<=q<=qmax.
    Replace with vendor‑fitted polynomials for production use.
    """
    if q_rel <= 0:
        return p0
    x = min(q_rel / max(q_rel_max, 1e-9), 1.0)
    return max(0.0, p0 * (1.0 - x * x))


def fan_w_rel(q_rel, w0, w2, q_rel_max):
    """Power coefficient curve: minimal convex rise with flow.
    w_rel = w0 + w2*(q/qmax)^2 . Replace with vendor fit if available.
    """
    if q_rel <= 0:
        return w0
    x = min(q_rel / max(q_rel_max, 1e-9), 1.0)
    return max(0.0, w0 + w2 * x * x)


# Example fan library (placeholders — substitute with Multi‑Wing/Cofimco/Moore data)
DEFAULT_FANS = {
    "Multi‑Wing H‑Series 9W": {
        "D0_m": 0.91, "n0_rpm": 960, "rho0": 1.2,
        "angles": {
            "25": {"q_rel_max": 0.14, "p0": 0.38, "w0": 0.10, "w2": 0.25},
            "30": {"q_rel_max": 0.16, "p0": 0.42, "w0": 0.12, "w2": 0.32},
            "35": {"q_rel_max": 0.17, "p0": 0.45, "w0": 0.14, "w2": 0.36}
        }
    },
    "Cofimco A‑Series": {
        "D0_m": 1.20, "n0_rpm": 740, "rho0": 1.2,
        "angles": {
            "20": {"q_rel_max": 0.12, "p0": 0.34, "w0": 0.09, "w2": 0.22},
            "28": {"q_rel_max": 0.145, "p0": 0.40, "w0": 0.11, "w2": 0.30},
            "35": {"q_rel_max": 0.165, "p0": 0.43, "w0": 0.13, "w2": 0.35}
        }
    },
    "Moore Axial": {
        "D0_m": 1.07, "n0_rpm": 900, "rho0": 1.2,
        "angles": {
            "22": {"q_rel_max": 0.13, "p0": 0.36, "w0": 0.10, "w2": 0.24},
            "30": {"q_rel_max": 0.155, "p0": 0.41, "w0": 0.12, "w2": 0.31},
            "38": {"q_rel_max": 0.17, "p0": 0.46, "w0": 0.15, "w2": 0.37}
        }
    }
}


def duty_point_axial(D_m, rpm, angle_params, rho_air, K_sys):
    """Find operating point where fan ΔP(Q) = K_sys * Q^2.
    Inputs:
      D_m: diameter [m], rpm: rotational speed [rpm]
      angle_params: dict with q_rel_max, p0, w0, w2
      rho_air: air density [kg/m³], K_sys: system coefficient [Pa/(m³/s)²]
    Returns dict with Q (m³/s), DP (Pa), Pshaft_kW, tip_speed (m/s)
    """
    n = max(rpm, 1e-6) / 60.0  # rps
    q_rel_max = angle_params["q_rel_max"]
    p0 = angle_params["p0"]
    w0 = angle_params["w0"]
    w2 = angle_params["w2"]

    Q_max = q_rel_max * n * (D_m ** 3)
    # Bisection between nearly zero and Q_max
    lo, hi = 0.0, max(Q_max, 1e-6)

    def f(Q):
        if Q <= 0:
            return fan_p_rel(0.0, p0, q_rel_max) * rho_air * (n * D_m) ** 2
        q_rel = Q / (n * (D_m ** 3))
        p_rel = fan_p_rel(q_rel, p0, q_rel_max)
        DP_fan = p_rel * rho_air * (n * D_m) ** 2
        DP_sys = K_sys * Q * Q
        return DP_fan - DP_sys

    # Ensure sign change by nudging hi if necessary
    f_lo = f(lo + 1e-9)
    f_hi = f(hi)
    if f_hi > 0:
        # Fan still overcomes system at free delivery: return Q_max
        Q_star = Q_max
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fm = f(mid)
            if fm > 0:
                lo = mid
            else:
                hi = mid
        Q_star = 0.5 * (lo + hi)

    q_rel_star = Q_star / (n * (D_m ** 3) + 1e-12)
    p_rel_star = fan_p_rel(q_rel_star, p0, q_rel_max)
    DP_star = p_rel_star * rho_air * (n * D_m) ** 2
    w_rel_star = fan_w_rel(q_rel_star, w0, w2, q_rel_max)
    Pshaft_W = w_rel_star * rho_air * (n ** 3) * (D_m ** 5)
    tip_speed = math.pi * D_m * n
    return {
        "Q_m3_s": Q_star,
        "DP_Pa": DP_star,
        "Pshaft_kW": Pshaft_W / 1000.0,
        "tip_speed_m_s": tip_speed,
    }

# Sidebar controls for fan modelling (separate block to avoid disturbing existing UI)
with st.sidebar:
    st.header("Fan modelling")
    fan_enabled = st.checkbox("Enable axial fan modelling", value=True)
    eff_total = st.number_input("Overall drive efficiency (motor×transmission)", 0.30, 0.95, 0.70, 0.01)
    # Editable fan library JSON
    fans_json = st.text_area(
        "Fan library (JSON — replace with vendor‑fitted values)",
        value=json.dumps(DEFAULT_FANS, indent=2),
        height=260,
    )
    try:
        fans_db = json.loads(fans_json)
    except Exception as e:
        st.error(f"Fan JSON parse error: {e}")
        fans_db = DEFAULT_FANS

# Compute system curve coefficient K_sys from current tower sizing point
# Let user add miscellaneous losses (drift eliminator, inlet louver, plenum)
if fan_enabled:
    st.markdown("## 🌀 Fan Duty Point (Axial)")
    DP_misc = st.number_input("Other system losses (Pa) — eliminators/inlet/etc.", 0.0, 500.0, 40.0, 1.0)
    DP_total_at_design = DP_fill + DP_misc
    K_sys = DP_total_at_design / max(Vdot_user ** 2, 1e-9)

    # Choose fan model and angle
    fan_name = st.selectbox("Fan model", list(fans_db.keys()))
    fan_info = fans_db[fan_name]
    angle_keys = list(fan_info["angles"].keys())
    angle_sel = st.selectbox("Blade angle (deg)", angle_keys, index=min(1, len(angle_keys)-1))
    angle_params = fan_info["angles"][angle_sel]

    # Mode: Manual or Auto‑size diameter
    mode_fan = st.radio("Mode", ["Manual fan", "Auto‑size diameter"], horizontal=True)

    fan_summary = None

    if mode_fan == "Manual fan":
        colF1, colF2, colF3 = st.columns(3)
        with colF1:
            D_m = st.number_input("Fan diameter (m)", 0.5, 6.0, float(fan_info.get("D0_m", 0.91)), 0.05)
        with colF2:
            rpm = st.number_input("Fan RPM", 200, 1800, int(fan_info.get("n0_rpm", 960)), 10)
        with colF3:
            tip_limit = st.number_input("Tip speed limit (m/s)", 30.0, 90.0, 65.0, 1.0)

        duty = duty_point_axial(D_m, rpm, angle_params, sugg["rho_mean"], K_sys)
        Q_fan = duty["Q_m3_s"]
        DP_fan = duty["DP_Pa"]
        Pshaft_kW = duty["Pshaft_kW"]
        tip_speed = duty["tip_speed_m_s"]
        motor_kW = Pshaft_kW / max(eff_total, 1e-6)

        fan_summary = {
            "fan": fan_name,
            "angle_deg": angle_sel,
            "D_m": D_m,
            "rpm": rpm,
            "Q_m3_s": Q_fan,
            "DP_Pa": DP_fan,
            "Pshaft_kW": Pshaft_kW,
            "motor_kW": motor_kW,
            "tip_m_s": tip_speed,
            "mode": "manual",
        }

        colM1, colM2, colM3, colM4 = st.columns(4)
        with colM1:
            st.metric("Fan duty flow (m³/s)", f"{Q_fan:.2f}")
        with colM2:
            st.metric("System ΔP at duty (Pa)", f"{K_sys * Q_fan * Q_fan:.0f}")
        with colM3:
            st.metric("Shaft power (kW)", f"{Pshaft_kW:.2f}")
        with colM4:
            st.metric("Motor kW (est)", f"{motor_kW:.2f}")
        st.caption(f"Tip speed: {tip_speed:.1f} m/s (limit {tip_limit:.0f} m/s)")
        if tip_speed > tip_limit:
            st.warning("Tip speed exceeds limit — consider larger diameter or lower RPM.")
        if abs(Q_fan - Vdot_user) / max(Vdot_user, 1e-9) > 0.1:
            st.info("Fan duty flow differs >10% from current design flow. Consider adjusting fan RPM/diameter or revisit fill area.")

        # Plot fan and system curves
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            n = rpm / 60.0
            Qs = np.linspace(0.0, max(Q_fan*1.3, 0.1), 80)
            q_rel = Qs / (n * (D_m ** 3) + 1e-12)
            p_rel = np.array([fan_p_rel(q, angle_params["p0"], angle_params["q_rel_max"]) for q in q_rel])
            DP_f = p_rel * sugg["rho_mean"] * (n * D_m) ** 2
            DP_sys_curve = K_sys * Qs * Qs
            fig = plt.figure()
            plt.plot(Qs, DP_f, label="Fan ΔP")
            plt.plot(Qs, DP_sys_curve, label="System ΔP")
            plt.scatter([Q_fan], [K_sys*Q_fan*Q_fan], marker='x')
            plt.xlabel("Flow Q (m³/s)")
            plt.ylabel("ΔP (Pa)")
            plt.legend()
            st.pyplot(fig)
        except Exception as e:
            st.caption(f"Curve plot unavailable: {e}")

    else:  # Auto‑size diameter
        target_Q = st.number_input("Target flow (m³/s)", 0.1, 500.0, float(Vdot_user), 0.1)
        max_rpm = st.number_input("Max RPM", 300, 1800, int(fan_info.get("n0_rpm", 960)), 10)
        tip_limit = st.number_input("Tip speed limit (m/s)", 30.0, 90.0, 65.0, 1.0)

        # Scan diameters to find smallest meeting target at max_rpm & tip limit
        D_best = None
        duty_best = None
        for D in [x/100 for x in range(60, 401, 5)]:  # 0.60 m to 4.00 m
            n = max_rpm / 60.0
            if math.pi * D * n > tip_limit:
                continue
            duty_try = duty_point_axial(D, max_rpm, angle_params, sugg["rho_mean"], K_sys)
            if duty_try["Q_m3_s"] >= target_Q:
                D_best = D
                duty_best = duty_try
                break
        if D_best is None:
            st.error("Target flow not achievable within RPM/tip limits using current angle curve. Try higher blade angle or relax limits.")
        else:
            # Refine: solve for RPM to hit target_Q with D_best (respect tip limit)
            rpm_lo, rpm_hi = 200, max_rpm
            for _ in range(40):
                rpm_mid = 0.5 * (rpm_lo + rpm_hi)
                if math.pi * D_best * (rpm_mid/60.0) > tip_limit:
                    rpm_hi = rpm_mid
                    continue
                q_mid = duty_point_axial(D_best, rpm_mid, angle_params, sugg["rho_mean"], K_sys)["Q_m3_s"]
                if q_mid < target_Q:
                    rpm_lo = rpm_mid
                else:
                    rpm_hi = rpm_mid
            rpm_req = rpm_hi
            duty = duty_point_axial(D_best, rpm_req, angle_params, sugg["rho_mean"], K_sys)
            Q_fan = duty["Q_m3_s"]
            DP_fan = duty["DP_Pa"]
            Pshaft_kW = duty["Pshaft_kW"]
            tip_speed = duty["tip_speed_m_s"]
            motor_kW = Pshaft_kW / max(eff_total, 1e-6)

            fan_summary = {
                "fan": fan_name,
                "angle_deg": angle_sel,
                "D_m": D_best,
                "rpm": rpm_req,
                "Q_m3_s": Q_fan,
                "DP_Pa": DP_fan,
                "Pshaft_kW": Pshaft_kW,
                "motor_kW": motor_kW,
                "tip_m_s": tip_speed,
                "mode": "autosize",
            }

            colA1, colA2, colA3, colA4 = st.columns(4)
            with colA1:
                st.metric("Suggested diameter (m)", f"{D_best:.2f}")
            with colA2:
                st.metric("RPM (est)", f"{rpm_req:.0f}")
            with colA3:
                st.metric("Flow at duty (m³/s)", f"{Q_fan:.2f}")
            with colA4:
                st.metric("Motor kW (est)", f"{motor_kW:.2f}")

    # Stash summary in session for PDF/reporting
    if 'fan_summary' not in st.session_state:
        st.session_state['fan_summary'] = None
    st.session_state['fan_summary'] = fan_summary

# -------------------------------
# PDF Report
# -------------------------------

def build_pdf_report() -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import mm

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    Wp, Hp = A4

    def line(y, text, size=10):
        c.setFont("Helvetica", size)
        c.drawString(20*mm, y, text)
        return y - 5*mm

    y = Hp - 20*mm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20*mm, y, f"Cooling Tower Sizing Report — {proj_name}")
    y -= 10*mm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20*mm, y, "Inputs")
    y -= 6*mm

    inputs = [
        ("Tower flow", flow_type),
        ("Fill", fill_name),
        ("Fill depth (m)", f"{depth_m:.2f}"),
        ("Free‑area frac", f"{free_area_frac:.2f}"),
        ("Water flow (L/min)", f"{Qw_L_min:.1f}"),
        ("Tin (°C)", f"{Tin_C:.2f}"),
        ("Tdb/WB (°C)", f"{Tdb_C:.1f} / {Twb_C:.1f}"),
        ("Sizing mode", mode),
        ("Tout target (°C)", f"{Tout_C:.2f}"),
        ("Heat load (kW)", f"{Q_kW:.1f}"),
        ("Fan flow (m³/s)", f"{Vdot_user:.2f}"),
        ("Target v_face (m/s)", f"{v_face_target:.2f}"),
    ]
    for k, v in inputs:
        y = line(y, f"• {k}: {v}")

    y -= 4*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20*mm, y, "Results")
    y -= 6*mm
    results = [
        ("Fill plan area (m²)", f"{A_fill:.2f}"),
        ("Air face velocity (m/s)", f"{v_air_face:.2f}"),
        ("Water superficial vel. (m/s)", f"{v_water:.3f}"),
        ("Water loading (m³/h·m²)", f"{water_loading_m3_h_m2:.1f}"),
        ("Fill ΔP (Pa)", f"{DP_fill:.0f}"),
        ("Achievable kW", f"{Q_ach_kW:.0f}"),
        ("Achievable Tout (°C)", f"{Tout_ach_C:.2f}"),
    ]
    for k, v in results:
        y = line(y, f"• {k}: {v}")

    y -= 4*mm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(20*mm, y, "Intermediate (psychrometrics)")
    y -= 6*mm
    interm = [
        ("W_in (kg/kg)", f"{W_in:.5f}"),
        ("h_in (kJ/kg_da)", f"{h_in:.2f}"),
        ("Δh (kJ/kg_da)", f"{delta_h:.2f}"),
        ("ρ_mean (kg/m³)", f"{rho_mean:.3f}"),
        ("Exhaust T assum. (°C)", f"{sugg['T_exh_C']:.2f}"),
    ]
    for k, v in interm:
        y = line(y, f"• {k}: {v}")

    # Fan section if available
    fan_summary = st.session_state.get('fan_summary', None)
    if fan_summary:
        y -= 4*mm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20*mm, y, "Fan (axial) duty")
        y -= 6*mm
        fan_items = [
            ("Model", fan_summary.get("fan")),
            ("Blade angle (deg)", fan_summary.get("angle_deg")),
            ("Diameter (m)", f"{fan_summary.get('D_m'):.2f}"),
            ("RPM", f"{fan_summary.get('rpm'):.0f}"),
            ("Flow (m³/s)", f"{fan_summary.get('Q_m3_s'):.2f}"),
            ("System ΔP (Pa)", f"{K_sys * fan_summary.get('Q_m3_s')**2:.0f}"),
            ("Shaft power (kW)", f"{fan_summary.get('Pshaft_kW'):.2f}"),
            ("Motor kW (est)", f"{fan_summary.get('motor_kW'):.2f}"),
            ("Tip speed (m/s)", f"{fan_summary.get('tip_m_s'):.1f}"),
        ]
        for k, v in fan_items:
            y = line(y, f"• {k}: {v}")

    c.showPage()
    c.save()
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

pdf_bytes = build_pdf_report()
st.download_button(
    label="📄 Download PDF report",
    data=pdf_bytes,
    file_name=f"{proj_name.replace(' ', '_')}_CoolingTowerReport.pdf",
    mime="application/pdf",
)

# -------------------------------
# Footnotes
# -------------------------------
st.caption(
    "This is a first‑pass estimator using moist‑air enthalpy balance to size airflow and fill plan area. Replace fill coefficients with vendor curves (e.g., Brentwood) and validate with Merkel/Poppe integration and test data before procurement."
)
