# Reliable under pressure

Interactive dashboard for a T20 pressure model built for SupeRR Selector Season 2, Round 1.

**Live dashboard:** https://possiblyneptune.github.io/T20-Pressure-Model/

It ranks IPL and T20I batters and bowlers (2022 to 2026) by how surely they beat a replacement-level player in their own role when the match is on the line. Pressure is measured on every ball as the swing in win probability the next delivery could cause (a Leverage Index).

The dashboard includes:

- the top-10 batters and bowlers with their 90% ranges
- ball-by-ball replays of six close finishes
- a player explorer covering all 131 eligible batters and 146 bowlers
- robustness results under 18 alternative settings

`index.html` is a single self-contained file: the data and fonts are embedded, so it also works offline.
