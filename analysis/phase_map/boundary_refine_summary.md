# Boundary Refinement Summary

## Coarse Boundary

- Coarse phase map places the r0 frontier between 1.00007 and 1.00010 among tested points.

## Dense Boundary

- At dt=90, success is first lost at r0_over_target=1.00004.
- Lowest final-radius-error success: dt `140`, r0_over_target `1.00003`, final_radius_error `7.205e+03`.

## Per-dt Frontier

- dt `80`: no successful dense-grid point
- dt `90`: success through 1.00003; first failure after success at 1.00004
- dt `100`: success through 1.00005; first failure after success at 1.00006
- dt `110`: no successful dense-grid point
- dt `120`: no successful dense-grid point
- dt `130`: success through 1.00008; first failure after success at 1.00009
- dt `140`: success through 1.00012; no failure after success in dense range