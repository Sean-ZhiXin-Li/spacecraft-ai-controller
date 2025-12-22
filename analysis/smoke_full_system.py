import json
from pathlib import Path

def check_spice() -> None:
    import spiceypy as sp
    print("[SPICE]", sp.tkvrsn("TOOLKIT"))

def check_casadi_ipopt() -> None:
    import casadi as ca
    x = ca.MX.sym("x")
    nlp = {"x": x, "f": (x - 1) ** 2}
    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
    }
    s = ca.nlpsol("s", "ipopt", nlp, opts)
    r = s(x0=0)
    print("[IPOPT] x* =", float(r["x"]))

def check_osqp() -> None:
    import osqp  # noqa: F401
    print("[OSQP] ok")

def check_do_mpc() -> None:
    import do_mpc
    print("[do-mpc]", do_mpc.__version__)

def check_orbit_init_json() -> None:
    p = Path("data/orbit_init_1au.json")
    d = json.loads(p.read_text(encoding="utf-8"))
    print("[OrbitInit] r_m =", d["r_m"])
    print("[OrbitInit] v_m_s =", d["v_m_s"])

def main() -> None:
    check_spice()
    check_casadi_ipopt()
    check_osqp()
    check_do_mpc()
    check_orbit_init_json()
    print("[SMOKE] full system OK")

if __name__ == "__main__":
    main()
