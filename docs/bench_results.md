# Benchmark of the potential-flow PINN on exact steady waves

Depth 3 m, wavelength 40 m; lidar-like samples: 40-m window at 5 cm, 5.17 scans per second, 5 mm noise; the window lasts 0.3 T (12 scans). Errors relative to the exact solution.

| run | H (m) | B exact | coverage | depth error (m) | stages | C error | u_crest error | B median | B range over scans | B error | surface u RMS / C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 6000, loose tolerances | -0.9% | -8.4% | 0.4005 | 0.393-0.444 | -0.0343 | 3.0% |
| b | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 15000, loose tolerances | -0.7% | -10.3% | 0.3883 | 0.379-0.409 | -0.0465 | 3.4% |
| c | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 6000, loose tolerances, sharper phi features | +3.5% | -51.7% | 0.2122 | 0.133-0.364 | -0.2226 | 12.6% |
| e | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 6000 | -0.3% | -6.9% | 0.4083 | 0.373-0.512 | -0.0265 | 2.1% |
| f | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 6000, crest focus | -1.9% | -9.9% | 0.4018 | 0.385-0.421 | -0.0330 | 3.2% |
| g | 1.5 | 0.4348 | 1.00 | 0.00 | Adam 6000 + L-BFGS 2000 | +0.3% | +0.1% | 0.4343 | 0.431-0.440 | -0.0005 | 0.1% |
| h | 1.5 | 0.4348 | 0.76 | 0.00 | Adam 6000 + L-BFGS 2000 | +0.5% | -0.1% | 0.4315 | 0.425-0.438 | -0.0033 | 0.1% |
| i | 1.75 | 0.5356 | 0.76 | 0.00 | Adam 6000 + L-BFGS 2000 | +0.2% | -0.2% | 0.5321 | 0.527-0.539 | -0.0035 | 0.1% |
| j | 1.5 | 0.4348 | 0.54 | 0.00 | Adam 6000 + L-BFGS 2000 | -0.2% | -0.3% | 0.4325 | 0.429-0.438 | -0.0023 | 0.2% |
| k | 1.5 | 0.4348 | 1.00 | 0.10 | Adam 6000 + L-BFGS 2000 | +0.1% | -1.0% | 0.4309 | 0.426-0.437 | -0.0039 | 0.3% |
| l | 1.5 | 0.4348 | 0.54 | 0.00 | Adam 6000 + L-BFGS 2000, trained on filled gaps | +0.3% | -0.3% | 0.4318 | 0.420-0.458 | -0.0030 | 0.2% |
