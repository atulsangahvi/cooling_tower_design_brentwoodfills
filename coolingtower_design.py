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

    if warns:
        y -= 4*mm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(20*mm, y, "Notes / Warnings")
        y -= 6*mm
        for w in warns:
            y = line(y, f"• {w}")

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
