# Retention-policy correction

Code commit: `7e95955ecab83037dec74dfe587d09189e0146fe`

Simulation training-range rejection and per-scene pixel-floor rejection removed.
All paired dates attempted; no coefficient interpolation or cross-date replacement.
Existing DN/reference/geometry arrays checked unchanged for 220 FY-3D and 133 FY-3F scene caches.
24 FY-3D and 22 FY-3F previously discarded small scenes rematched from real inputs.
25 unit tests passed. Main-entry tests cover one fitted day and one day with no valid samples.
Rendering and QA completed on node111/node112.
Independent-date comparisons freeze original train/test dates and use identical expanded held-out samples for native, old-model and new-model predictions.
