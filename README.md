# portfolio-analytics
Monthly attribution of three First Eagle funds (“Global,” “Overseas,” “US”) versus a 20-factor macro/style library.
ch Stack
Python 3.11 with pandas, numpy, statsmodels, yfinance

Logging for run-time diagnostics

CI/CD Friendly (module structure + config file)

Core Workflow
Data Ingestion & Prep

Downloads max-history prices (Equity, Rates, Credit, Commodities, FX, Trend, Volatility, etc.) via yfinance.

Monthly resampling (MS) and auto-ffill date alignment.

Excess‐Return Calculation

Fetches 3-month T-Bill yield (^IRX), converts to monthly, aligns to factors.

Computes fund excess returns over T-Bill.

Static OLS Attribution (Full Sample)

Runs cross-sectional OLS per fund to estimate α, R², and β coefficients for all 20 factors.

Outputs first_eagle_static_multifactor_betas.csv.

Rolling Window OLS (Research-Grade)

Custom loop for a 36-month rolling regression, handling NaNs gracefully.

Concatenates rolling betas into a unified DataFrame and exports first_eagle_rolling_multifactor_betas.csv.

Snapshot Last 36-Month Attribution

Static OLS on most recent 36 months for “as-of” factor exposures.

Impact & Next Steps
Automated Reporting: From 4 hrs of manual work down to <10 min automatic runs.

Extensible: Swap in new factors or funds via simple config changes.

Scalable: Handles 3–300 tickers without code changes.

Upcoming Enhancements: Add risk‐adjusted metrics (tracking error, Information Ratio) and CLI flags for live vs. backtest modes.
