import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ----- CONFIG -----
START = "2000-01-01"
FUNDS = {
    "Global Fund":   "SGIIX",
    "Overseas Fund": "SGOIX",
    "US Fund":       "FEVIX",
}

# ----- 1) BUILD FACTOR PROXIES via YFinance -----
factor_tickers = [
    "SPY","TLT","HYG","DBC","EEM","UUP","TIP",
    "SVXY","SHY","CWY","USMV","MTUM","QUAL","IVE","IWM","ACWI",
    "GLD","USO","VIXY"  # New macro factors: Gold, Oil, Volatility
]

def download_prices(tickers):
    dfs = []
    for t in tickers:
        try:
            df = yf.download(t, period="max", auto_adjust=False, progress=False)
            if df.empty:
                print(f"Warning: No data for {t}")
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            frame = df[[col]].rename(columns={col: t})
            dfs.append(frame)
        except Exception as e:
            print(f"Error downloading {t}: {e}")
    if not dfs:
        return pd.DataFrame()
    prices = pd.concat(dfs, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices.columns.name = None
    print("Tickers with data:", list(prices.columns))
    return prices

price_df = download_prices(factor_tickers).resample("MS").last()

# Exclude incomplete current month
today = pd.Timestamp.today().normalize()
price_df = price_df[price_df.index < today.replace(day=1)]

raw_rets = price_df.pct_change().dropna()

rename_map = {
    "SPY":  "Equity",
    "TLT":  "Interest Rates",
    "HYG":  "Credit",
    "DBC":  "Commodities",
    "EEM":  "Emerging Markets",
    "UUP":  "FX",
    "TIP":  "Real Yields",
    "SVXY": "Equity Short Vol",
    "CWY":  "FX Carry",  # TODO: Replace with your own FX carry series if available
    "USMV": "Low Risk",
    "MTUM": "Momentum",
    "QUAL": "Quality",
    "IVE":  "Value",
    "IWM":  "Small Cap",
    "GLD":  "Gold",
    "USO":  "Oil",
    "VIXY": "Volatility"
    # ACWI intentionally not renamed
}
f = raw_rets.rename(columns=rename_map)

# Local Equity (US-based)
f["Local Equity"] = f["Equity"] - raw_rets["ACWI"]
f["Local Inflation"] = raw_rets['TIP'] - raw_rets['TLT']

# Carry & Trend factors
f["FI Carry"] = raw_rets["TLT"] - raw_rets["SHY"]
f["Trend"] = price_df["SPY"].pct_change(12)


# Final factor list
factor_cols = [
    "Equity","Interest Rates","Credit","Commodities",
    "Emerging Markets","FX","Real Yields","Local Inflation","Local Equity",
    "Equity Short Vol","FI Carry","FX Carry","Trend",
    "Low Risk","Momentum","Quality","Value","Small Cap",
    "Gold","Oil","Volatility"
]
# Remove any missing columns (if a download failed)
factor_cols = [col for col in factor_cols if col in f.columns]
factors = f[factor_cols]

# ----- DEBUG: Show NaN counts in each factor -----
print("\n[DEBUG] NaN counts in each factor before dropna:")
print(factors.isna().sum())

# ----- 2) RISK-FREE RATE (3M T-Bill, monthly) -----
try:
    rf_raw = yf.download('^IRX', start=START, progress=False)['Close']
    rf = rf_raw / 1200  # ^IRX is annualized %; convert to monthly decimal
    rf = rf.reindex(factors.index, method='ffill').fillna(0)
except Exception as e:
    print("Risk-free rate download failed, using 0% RF.")
    rf = pd.Series(0.0, index=factors.index, name="RF")

if isinstance(rf, pd.DataFrame):
    if 'Close' in rf.columns:
        rf = rf['Close']
    else:
        rf = rf.iloc[:, 0]
    rf = rf.squeeze()

# ----- 3) DOWNLOAD FUND RETURNS -----
fund_prices = download_prices(list(FUNDS.values())).resample("MS").last()
fund_prices = fund_prices[fund_prices.index < today.replace(day=1)]
fund_rets   = fund_prices.pct_change().dropna()
fund_rets.columns = list(FUNDS.keys())

# ----- 4) MERGE & EXCESS RETURNS -----
df = fund_rets.join(factors, how="outer").ffill().dropna()
rf_aligned = rf.reindex(df.index, method='ffill').astype(float)
for fund in fund_rets:
    df[f"{fund}_Excess"] = df[fund] - rf_aligned

print("\n[DEBUG] Shape of merged df after dropna:", df.shape)
print("[DEBUG] First date in df:", df.index.min(), "Last date:", df.index.max())
print("[DEBUG] NaN counts for all columns in merged df:")
print(df.isna().sum())

# ----- 5) STATIC OLS -----
static_results = []
for fund in fund_rets:
    y = df[f"{fund}_Excess"]
    X = sm.add_constant(df[factor_cols])
    m = sm.OLS(y, X).fit()
    row = {"Fund": fund, "R2": m.rsquared, "Alpha": m.params["const"]}
    for col in factor_cols:
        row[f"β {col}"] = m.params.get(col, np.nan)
    static_results.append(row)

static_df = pd.DataFrame(static_results).set_index("Fund").round(3)
print("\n=== STATIC MULTI-FACTOR EXPOSURES (FULL SAMPLE) ===")
print(static_df)
static_df.to_csv("first_eagle_static_multifactor_betas.csv")

# ----- 6) CUSTOM ROLLING OLS (robust, research-grade) -----
def rolling_ols_loop(y, X, window):
    betas = []
    idxs = []
    for i in range(window-1, len(y)):
        y_window = y.iloc[i-window+1:i+1]
        X_window = X.iloc[i-window+1:i+1]
        if y_window.isna().any() or X_window.isna().any().any():
            betas.append([np.nan]*X.shape[1])
        else:
            model = sm.OLS(y_window, X_window).fit()
            betas.append(model.params.values)
        idxs.append(y.index[i])
    return pd.DataFrame(betas, index=idxs, columns=X.columns)

window = 36  # Adjust as desired
rolling_betas_all = []

for fund in fund_rets:
    needed_cols = [f"{fund}_Excess"] + factor_cols
    df_fund = df[needed_cols].dropna()
    print(f"\n[DEBUG] {fund} - rows after dropna: {len(df_fund)}")
    print(f"[DEBUG] {fund} - first date: {df_fund.index.min()}, last date: {df_fund.index.max()}")
    if len(df_fund) >= window:
        y = df_fund[f"{fund}_Excess"]
        X = sm.add_constant(df_fund[factor_cols])
        rolling_betas = rolling_ols_loop(y, X, window)
        rolling_betas = rolling_betas.add_prefix(f"{fund}_")
        rolling_betas_all.append(rolling_betas)
        print(f"[DEBUG] {fund} rolling betas (last 5):")
        print(rolling_betas.tail())
    else:
        print(f"Not enough data for rolling regression for {fund} (need at least {window} rows)")

if rolling_betas_all:
    rolling_betas_df = pd.concat(rolling_betas_all, axis=1)
    print(f"\n[DEBUG] Number of non-NaN rows in rolling_betas_df:", rolling_betas_df.dropna().shape[0])
    print("[DEBUG] First 10 rows of rolling_betas_df:")
    print(rolling_betas_df.head(10))
    print("[DEBUG] Last 10 rows of rolling_betas_df:")
    print(rolling_betas_df.tail(10))

    if rolling_betas_df.dropna().shape[0] > 0:
        last_valid_idx = rolling_betas_df.dropna().index[-1]
        print("\n=== Last valid rolling betas for all funds (your 'current' rolling attribution) ===")
        print(rolling_betas_df.loc[last_valid_idx])
    else:
        print("\nNo valid rolling betas (all windows contain NaNs). Try reducing the window size or checking for missing data in factors.")

    rolling_betas_df.to_csv("first_eagle_rolling_multifactor_betas.csv")
else:
    print("\nNo rolling betas calculated (not enough data).")

# ----- 7) STATIC OLS ON LAST 36 MONTHS (for "as of last date" attribution) -----
print("\n=== STATIC MULTI-FACTOR EXPOSURES (LAST 36 MONTHS) ===")
for fund in fund_rets:
    needed_cols = [f"{fund}_Excess"] + factor_cols
    df_recent = df[needed_cols].dropna().iloc[-window:]
    if len(df_recent) == window:
        y = df_recent[f"{fund}_Excess"]
        X = sm.add_constant(df_recent[factor_cols])
        m = sm.OLS(y, X).fit()
        print(f"\n{fund} (as of {df_recent.index[-1].date()}):")
        print(m.params.round(3))
    else:
        print(f"\nNot enough data for static OLS on last {window} months for {fund}.")
