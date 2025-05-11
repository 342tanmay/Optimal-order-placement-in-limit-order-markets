# Optimal-order-placement-in-limit-order-markets

## Overview
The notebook **backtest.ipynb** runs a simple smart‑order‑router (SOR) back‑test on top‑of‑book quote data. It simulates splitting a parent buy order across several exchanges and compares the achieved price with a few baselines.

My work was done in a Jupyter notebook for cell-by-cell testing convenience, but a Python file will be included in the Repo to make sure it follows the task guidelines.

## Code layout
1. **Setup** – installs packages, imports libraries, and loads `l1_day.csv` into a pandas DataFrame.  
2. **Data model** – a `Venue` record holds the ask price, available size, taker fee, and maker rebate for each exchange.  
3. **Cost model** – `compute_cost` measures how expensive a proposed split is.  
4. **Allocation search** – `allocate` tries every way to divide the shares (in 100‑share chunks) across venues and keeps the cheapest.  
5. **Router loop** – `run_router` steps through market snapshots, calls `allocate`, simulates fills, and updates cash and remaining size.  
6. **Parameter tuning** – `tune_router` loops over five candidate values for each of the three penalty weights (`lambda_over`, `lambda_under`, `theta`), making 5 × 5 × 5 = 125 combinations, and keeps the one with the lowest average execution price.  
7. **Baselines** – helper functions that implement “always hit the best ask”, TWAP, and VWAP for comparison.  
8. **Reporting** – the last cell writes a JSON summary and draws a bar chart to compare methods.

## Search choices
- Grid search for penalty weights is straightforward but slow if more knobs are added.  
- Allocation search time grows quickly with more venues or finer share increments. Using a 100‑share step keeps it manageable.

## Suggested improvements
- Replace the brute‑force split with an integer linear program or dynamic program.  
- Swap the grid search for Bayesian optimization.
- Add a simple fill‑probability model so maker orders sometimes fail to trade.
