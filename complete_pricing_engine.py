"""
Quantitative Commodity Pricing Engine - Complete Implementation
Modules 1-4: Data Ingestion, Monte Carlo Simulation, Option Pricing, and Risk Analysis

Author: Quantitative Analysis Team
Date: December 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODULE 1: DATA INGESTION & PARAMETER ESTIMATION
# ============================================================================

class CommodityData:
    """
    A class to handle commodity futures data download, processing, and analysis.
    
    Attributes:
        ticker (str): The ticker symbol for the commodity
        data (pd.DataFrame): Historical price data
        log_returns (pd.Series): Calculated log returns
        annualized_volatility (float): Annualized volatility estimate
        mean_reversion_level (float): Long-term mean price level
    """
    
    def __init__(self, ticker: str, years: int = 5):
        """
        Initialize the CommodityData object.
        
        Args:
            ticker: Yahoo Finance ticker symbol (e.g., 'CL=F' for WTI Crude)
            years: Number of years of historical data to download
        """
        self.ticker = ticker
        self.years = years
        self.data = None
        self.log_returns = None
        self.annualized_volatility = None
        self.mean_reversion_level = None
        
    def download_data(self, use_simulation: bool = False) -> pd.DataFrame:
        """
        Download historical data from Yahoo Finance, or use simulated data.
        
        Args:
            use_simulation: If True, generate realistic simulated data
            
        Returns:
            DataFrame containing historical price data
        """
        if use_simulation:
            print(f"Generating simulated {self.years}-year data for {self.ticker}...")
            self.data = self._generate_simulated_data()
            print(f"Successfully generated {len(self.data)} data points\n")
            return self.data
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.years)
        
        print(f"Downloading {self.years}-year historical data for {self.ticker}...")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        
        try:
            self.data = yf.download(
                self.ticker,
                start=start_date,
                end=end_date,
                progress=False
            )
            
            if self.data.empty:
                raise ValueError(f"No data downloaded for {self.ticker}")
            
            print(f"Successfully downloaded {len(self.data)} data points\n")
            return self.data
        except Exception as e:
            print(f"Download failed: {str(e)[:100]}")
            print("Falling back to simulated data...\n")
            return self.download_data(use_simulation=True)
    
    def _generate_simulated_data(self) -> pd.DataFrame:
        """
        Generate realistic simulated WTI Crude Oil price data.
        Uses Geometric Brownian Motion with mean reversion characteristics.
        
        Returns:
            DataFrame containing simulated price data
        """
        # Parameters based on historical WTI characteristics
        S0 = 70.0  # Initial price
        mu = 0.0001  # Drift (slight upward trend)
        sigma = 0.025  # Daily volatility (~40% annualized)
        theta = 0.15  # Mean reversion speed
        mean_price = 75.0  # Long-term mean price
        
        # Generate dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.years)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        n_days = len(dates)
        
        # Generate price path with mean reversion
        np.random.seed(42)  # For reproducibility
        prices = np.zeros(n_days)
        prices[0] = S0
        
        for t in range(1, n_days):
            # Mean reversion + GBM
            drift = theta * (mean_price - prices[t-1]) / prices[t-1]
            shock = sigma * np.random.randn()
            prices[t] = prices[t-1] * np.exp(drift + shock)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(n_days) * 0.002),
            'High': prices * (1 + np.abs(np.random.randn(n_days)) * 0.01),
            'Low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.01),
            'Close': prices,
            'Volume': np.random.randint(100000, 500000, n_days)
        }, index=dates)
        
        # Ensure High is highest and Low is lowest
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
        
        return df
    
    def calculate_log_returns(self) -> pd.Series:
        """Calculate daily log returns from adjusted close prices."""
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")
        
        prices = self.data['Close']
        self.log_returns = np.log(prices / prices.shift(1))
        self.log_returns = self.log_returns.dropna()
        
        return self.log_returns
    
    def calculate_annualized_volatility(self, trading_days: int = 252) -> float:
        """Calculate annualized volatility from daily log returns."""
        if self.log_returns is None:
            raise ValueError("Log returns not calculated. Call calculate_log_returns() first.")
        
        daily_volatility = self.log_returns.std()
        self.annualized_volatility = daily_volatility * np.sqrt(trading_days)
        
        return self.annualized_volatility
    
    def calculate_mean_reversion_level(self) -> float:
        """Calculate the long-term historical mean price (mean reversion level)."""
        if self.data is None:
            raise ValueError("Data not loaded. Call download_data() first.")
        
        self.mean_reversion_level = self.data['Close'].mean()
        return self.mean_reversion_level
    
    def analyze(self, use_simulation: bool = False) -> dict:
        """Perform complete analysis: download data and calculate all parameters."""
        self.download_data(use_simulation=use_simulation)
        self.calculate_log_returns()
        self.calculate_annualized_volatility()
        self.calculate_mean_reversion_level()
        
        results = {
            'ticker': self.ticker,
            'data_points': len(self.data),
            'start_date': self.data.index[0],
            'end_date': self.data.index[-1],
            'current_price': self.data['Close'].iloc[-1],
            'mean_reversion_level': self.mean_reversion_level,
            'annualized_volatility': self.annualized_volatility,
            'volatility_percentage': self.annualized_volatility * 100,
            'min_price': self.data['Close'].min(),
            'max_price': self.data['Close'].max(),
        }
        
        return results
    
    def print_summary(self):
        """Print a formatted summary of the analysis results."""
        if self.data is None:
            raise ValueError("Analysis not performed. Call analyze() first.")
        
        print("=" * 70)
        print("MODULE 1: PARAMETER ESTIMATION RESULTS")
        print("=" * 70)
        print(f"\nCommodity Ticker: {self.ticker}")
        print(f"Analysis Period: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"Total Data Points: {len(self.data)}")
        print(f"\n{'PRICE STATISTICS':^70}")
        print("-" * 70)
        print(f"Current Price (S0):         ${self.data['Close'].iloc[-1]:.2f}")
        print(f"Mean Reversion Level (μ):   ${self.mean_reversion_level:.2f}")
        print(f"Historical Minimum:         ${self.data['Close'].min():.2f}")
        print(f"Historical Maximum:         ${self.data['Close'].max():.2f}")
        print(f"\n{'VOLATILITY METRICS':^70}")
        print("-" * 70)
        print(f"Annualized Volatility (σ):  {self.annualized_volatility:.4f} ({self.annualized_volatility * 100:.2f}%)")
        print(f"Daily Volatility:           {self.log_returns.std():.4f}")
        print("=" * 70)
        print()


# ============================================================================
# MODULE 2: MONTE CARLO SIMULATION ENGINE (ORNSTEIN-UHLENBECK PROCESS)
# ============================================================================

class MonteCarloEngine:
    """
    Monte Carlo simulation engine for commodity price paths using the
    Ornstein-Uhlenbeck (OU) mean-reverting process.
    
    The OU process is ideal for commodities because:
    1. Prices tend to revert to a long-term equilibrium (supply/demand balance)
    2. Unlike stocks, commodities don't have unlimited upside
    3. Physical constraints and storage costs create mean reversion
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo Engine.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.random_seed = random_seed
        self.simulated_paths = None
        self.time_grid = None
        
    def simulate_ou_process(
        self,
        S0: float,
        mu: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_simulations: int
    ) -> np.ndarray:
        """
        Simulate commodity price paths using the Ornstein-Uhlenbeck process.
        
        MATHEMATICAL FOUNDATION:
        ========================
        The Ornstein-Uhlenbeck SDE (Stochastic Differential Equation):
        
            dX_t = θ(μ - X_t)dt + σ dW_t
        
        Where:
            X_t = Current price at time t
            μ (mu) = Long-term mean (equilibrium price)
            θ (theta) = Mean reversion speed (how fast it reverts to μ)
            σ (sigma) = Volatility (randomness/uncertainty)
            dW_t = Wiener process (Brownian motion) increment
        
        INTERPRETATION:
        ===============
        - θ(μ - X_t)dt: DRIFT TERM (deterministic)
          * If X_t > μ: drift is negative → price pulled DOWN toward mean
          * If X_t < μ: drift is positive → price pulled UP toward mean
          * θ controls the strength: larger θ = faster mean reversion
        
        - σ dW_t: DIFFUSION TERM (stochastic/random)
          * Random shocks from supply/demand, geopolitics, weather, etc.
          * σ controls magnitude of randomness
        
        DISCRETE APPROXIMATION (Euler-Maruyama Method):
        ================================================
        We discretize the continuous SDE for numerical simulation:
        
            X_{t+Δt} = X_t + θ(μ - X_t)Δt + σ√(Δt) * Z
        
        Where:
            Δt = T / n_steps (time step size)
            Z ~ N(0,1) (standard normal random variable)
            √(Δt) = scales the random shock to the time interval
        
        Args:
            S0: Current spot price (starting point for all paths)
            mu: Long-term mean price (equilibrium level)
            theta: Mean reversion speed (0.1 = slow, 0.5 = fast)
            sigma: Annualized volatility (standard deviation)
            T: Time horizon in years (e.g., 1.0 for one year)
            n_steps: Number of time steps (e.g., 252 for daily steps)
            n_simulations: Number of Monte Carlo paths to simulate
            
        Returns:
            np.ndarray: Shape (n_simulations, n_steps + 1)
                       Each row is one simulated price path
                       Columns represent time steps from 0 to T
        """
        # Set random seed for reproducibility
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Time discretization
        dt = T / n_steps  # Time step size (e.g., 1/252 ≈ 0.004 for daily)
        sqrt_dt = np.sqrt(dt)  # Precompute for efficiency
        
        # Create time grid for plotting
        self.time_grid = np.linspace(0, T, n_steps + 1)
        
        # Initialize price matrix: rows = simulations, columns = time steps
        # Shape: (n_simulations, n_steps + 1)
        prices = np.zeros((n_simulations, n_steps + 1))
        
        # Set initial price for all paths
        prices[:, 0] = S0
        
        # MAIN SIMULATION LOOP
        # ====================
        # Simulate each time step for all paths simultaneously (vectorized)
        for t in range(n_steps):
            # Generate random shocks for all simulations at this time step
            # Z ~ N(0,1): standard normal random variables
            # Shape: (n_simulations,)
            Z = np.random.randn(n_simulations)
            
            # Ornstein-Uhlenbeck update formula (Euler-Maruyama discretization):
            # 
            # X_{t+1} = X_t + θ(μ - X_t)Δt + σ√(Δt)Z
            #           ^^^^   ^^^^^^^^^^^   ^^^^^^^^^
            #         current  drift term   diffusion term
            #         price   (mean rev)    (randomness)
            
            # Decomposed for clarity:
            current_prices = prices[:, t]                      # X_t
            drift = theta * (mu - current_prices) * dt         # θ(μ - X_t)Δt
            diffusion = sigma * sqrt_dt * Z                    # σ√(Δt)Z
            
            # Update prices for next time step
            prices[:, t + 1] = current_prices + drift + diffusion
            
            # OPTIONAL: Enforce non-negativity (commodities can't have negative prices)
            # Uncomment if needed for highly volatile scenarios:
            # prices[:, t + 1] = np.maximum(prices[:, t + 1], 0.01)
        
        # Store results
        self.simulated_paths = prices
        
        return prices
    
    def get_terminal_prices(self) -> np.ndarray:
        """
        Get the final prices at maturity for all simulations.
        
        Returns:
            np.ndarray: Terminal prices (last column of simulated paths)
        """
        if self.simulated_paths is None:
            raise ValueError("No simulations run yet. Call simulate_ou_process() first.")
        
        return self.simulated_paths[:, -1]
    
    def print_simulation_summary(self):
        """Print summary statistics of the simulation."""
        if self.simulated_paths is None:
            raise ValueError("No simulations run yet. Call simulate_ou_process() first.")
        
        terminal_prices = self.get_terminal_prices()
        
        print("=" * 70)
        print("MODULE 2: MONTE CARLO SIMULATION RESULTS")
        print("=" * 70)
        print(f"\nSimulation Parameters:")
        print(f"  Number of Paths:     {self.simulated_paths.shape[0]:,}")
        print(f"  Time Steps:          {self.simulated_paths.shape[1] - 1:,}")
        print(f"  Initial Price:       ${self.simulated_paths[0, 0]:.2f}")
        print(f"\nTerminal Price Statistics (at Maturity):")
        print(f"  Mean:                ${terminal_prices.mean():.2f}")
        print(f"  Median:              ${np.median(terminal_prices):.2f}")
        print(f"  Std Dev:             ${terminal_prices.std():.2f}")
        print(f"  Min:                 ${terminal_prices.min():.2f}")
        print(f"  Max:                 ${terminal_prices.max():.2f}")
        print(f"  5th Percentile:      ${np.percentile(terminal_prices, 5):.2f}")
        print(f"  95th Percentile:     ${np.percentile(terminal_prices, 95):.2f}")
        print("=" * 70)
        print()


# ============================================================================
# MODULE 3: OPTION PRICING (ASIAN CALL OPTION)
# ============================================================================

class OptionPricer:
    """
    Option pricing engine for exotic derivatives on commodities.
    
    ASIAN OPTIONS:
    ==============
    An Asian option's payoff depends on the AVERAGE price over the option's life,
    not just the terminal price. This makes them ideal for:
    
    1. AIRLINES: Want to hedge average fuel cost over a quarter/year
    2. REFINERIES: Need stable input costs averaged over production cycles
    3. MANUFACTURERS: Budget based on average commodity costs
    
    ADVANTAGES:
    ===========
    - Less volatile than standard options (averaging smooths price swings)
    - Harder to manipulate (can't game the price at expiry)
    - Cheaper premiums (lower volatility = lower price)
    - Better match for businesses with continuous commodity exposure
    """
    
    def __init__(self, monte_carlo_engine: MonteCarloEngine):
        """
        Initialize the Option Pricer.
        
        Args:
            monte_carlo_engine: Fitted MonteCarloEngine with simulated paths
        """
        self.mc_engine = monte_carlo_engine
        self.option_price = None
        self.payoffs = None
        
    def price_asian_call(
        self,
        strike: float,
        risk_free_rate: float,
        T: float
    ) -> float:
        """
        Price an Asian Call Option using Monte Carlo simulation.
        
        ASIAN CALL OPTION PAYOFF:
        =========================
        For each simulated price path:
        
        1. Calculate the ARITHMETIC AVERAGE price over the path:
           A = (1/n) * Σ S_i  where i = 0, 1, ..., n
        
        2. Calculate payoff at maturity:
           Payoff = max(A - K, 0)
           
           Where:
           - A = average price over the option's life
           - K = strike price (contracted price)
           
        3. If A > K: Option is "in the money" → Payoff = A - K
           If A ≤ K: Option is "out of the money" → Payoff = 0
        
        EXAMPLE:
        ========
        Strike = $75, Quarterly fuel contract
        
        Path 1: Prices = [$72, $76, $78, $74] → Average = $75.00
                Payoff = max($75 - $75, 0) = $0 (at the money)
        
        Path 2: Prices = [$70, $72, $76, $82] → Average = $75.00
                Payoff = max($75 - $75, 0) = $0 (at the money)
        
        Path 3: Prices = [$78, $80, $82, $84] → Average = $81.00
                Payoff = max($81 - $75, 0) = $6 (in the money!)
        
        DISCOUNTING:
        ============
        Future payoffs must be discounted to present value:
        
            PV = FV * e^(-r*T)
        
        Where:
            r = risk-free rate (e.g., Treasury yield)
            T = time to maturity in years
        
        FINAL OPTION PRICE:
        ===================
            C = e^(-rT) * E[max(A - K, 0)]
        
        Where E[·] is the expected value (mean) over all simulations.
        
        Args:
            strike: Strike price (K) - the contracted/guaranteed price
            risk_free_rate: Annual risk-free rate (e.g., 0.042 for 4.2%)
            T: Time to maturity in years
            
        Returns:
            float: Present value of the Asian call option
        """
        if self.mc_engine.simulated_paths is None:
            raise ValueError("No simulations available. Run Monte Carlo first.")
        
        # Get all simulated price paths
        paths = self.mc_engine.simulated_paths  # Shape: (n_sims, n_steps + 1)
        
        # STEP 1: Calculate arithmetic average for each path
        # ===================================================
        # axis=1 means average across columns (time steps) for each row (simulation)
        average_prices = np.mean(paths, axis=1)  # Shape: (n_sims,)
        
        # STEP 2: Calculate payoff for each path
        # ========================================
        # Vectorized: applies max() element-wise to entire array
        self.payoffs = np.maximum(average_prices - strike, 0)  # Shape: (n_sims,)
        
        # STEP 3: Calculate expected payoff (mean across all simulations)
        # =================================================================
        expected_payoff = np.mean(self.payoffs)
        
        # STEP 4: Discount back to present value
        # ========================================
        # e^(-rT) is the discount factor
        discount_factor = np.exp(-risk_free_rate * T)
        self.option_price = discount_factor * expected_payoff
        
        return self.option_price
    
    def calculate_delta(
        self,
        S0: float,
        mu: float,
        theta: float,
        sigma: float,
        T: float,
        n_steps: int,
        n_simulations: int,
        strike: float,
        risk_free_rate: float,
        bump_size: float = 1.0
    ) -> float:
        """
        Calculate Delta: sensitivity of option price to spot price changes.
        
        DELTA (Δ):
        ==========
        Delta measures how much the option price changes when the underlying
        spot price changes by $1.
        
            Δ ≈ (C(S0 + h) - C(S0)) / h
        
        Where:
            C(S0) = option price at current spot
            C(S0 + h) = option price at spot + bump
            h = bump size (typically $1)
        
        INTERPRETATION:
        ===============
        - Δ = 0.50: If spot ↑ $1, option price ↑ $0.50
        - Δ = 0.00-1.00 for calls (0 = far out of money, 1 = deep in money)
        - Used for DELTA HEDGING: to remain market neutral
        
        EXAMPLE:
        ========
        Current Price: $75, Option Price: $5.20, Delta: 0.52
        
        If crude rises to $76:
            New Option Price ≈ $5.20 + 0.52 × $1 = $5.72
        
        HEDGING APPLICATION:
        ====================
        If you BUY 1 call option (Δ = 0.52):
            → SHORT 0.52 futures to hedge
            → If spot ↑ $1: option gains $0.52, futures lose $0.52 → Net = $0
        
        Args:
            S0: Current spot price
            mu: Long-term mean
            theta: Mean reversion speed
            sigma: Volatility
            T: Time to maturity
            n_steps: Number of time steps
            n_simulations: Number of simulations
            strike: Strike price
            risk_free_rate: Risk-free rate
            bump_size: Size of price bump for finite difference (default: $1)
            
        Returns:
            float: Delta (option price sensitivity to spot price)
        """
        # Price at current spot (already calculated)
        price_at_S0 = self.option_price
        
        # Price at bumped spot (S0 + bump)
        mc_bumped = MonteCarloEngine(random_seed=self.mc_engine.random_seed)
        mc_bumped.simulate_ou_process(
            S0=S0 + bump_size,
            mu=mu,
            theta=theta,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            n_simulations=n_simulations
        )
        
        pricer_bumped = OptionPricer(mc_bumped)
        price_at_S0_plus_bump = pricer_bumped.price_asian_call(strike, risk_free_rate, T)
        
        # Calculate Delta using finite difference
        delta = (price_at_S0_plus_bump - price_at_S0) / bump_size
        
        return delta
    
    def print_pricing_summary(self, strike: float, delta: Optional[float] = None):
        """Print summary of option pricing results."""
        if self.option_price is None:
            raise ValueError("Option not priced yet. Call price_asian_call() first.")
        
        print("=" * 70)
        print("MODULE 3: ASIAN CALL OPTION PRICING RESULTS")
        print("=" * 70)
        print(f"\nOption Specification:")
        print(f"  Type:                Asian Call (Arithmetic Average)")
        print(f"  Strike Price (K):    ${strike:.2f}")
        print(f"  Moneyness:           {'ATM' if abs(self.mc_engine.simulated_paths[0, 0] - strike) < 1 else 'ITM' if self.mc_engine.simulated_paths[0, 0] > strike else 'OTM'}")
        print(f"\nPricing Results:")
        print(f"  Option Price:        ${self.option_price:.4f}")
        print(f"  Expected Payoff:     ${np.mean(self.payoffs):.4f}")
        print(f"  Payoff Std Dev:      ${np.std(self.payoffs):.4f}")
        print(f"  Max Payoff:          ${np.max(self.payoffs):.4f}")
        print(f"  % In-The-Money:      {100 * np.sum(self.payoffs > 0) / len(self.payoffs):.2f}%")
        
        if delta is not None:
            print(f"\nGreeks:")
            print(f"  Delta (Δ):           {delta:.4f}")
            print(f"  Interpretation:      ${1:.2f} spot ↑ → ${delta:.4f} option ↑")
        
        print("=" * 70)
        print()


# ============================================================================
# MODULE 4: RISK ANALYSIS & VISUALIZATION
# ============================================================================

class RiskAnalyzer:
    """
    Risk analysis and visualization toolkit for commodity derivatives.
    """
    
    def __init__(
        self,
        commodity_data: CommodityData,
        mc_engine: MonteCarloEngine,
        option_pricer: OptionPricer
    ):
        """
        Initialize the Risk Analyzer.
        
        Args:
            commodity_data: Fitted CommodityData object
            mc_engine: Fitted MonteCarloEngine object
            option_pricer: Fitted OptionPricer object
        """
        self.commodity_data = commodity_data
        self.mc_engine = mc_engine
        self.option_pricer = option_pricer
    
    def plot_price_paths(
        self,
        n_paths: int = 50,
        mu: Optional[float] = None,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize simulated price paths with mean reversion overlay.
        
        Args:
            n_paths: Number of random paths to display
            mu: Long-term mean to overlay (if None, uses commodity_data mean)
            figsize: Figure size
        """
        if self.mc_engine.simulated_paths is None:
            raise ValueError("No simulations available.")
        
        paths = self.mc_engine.simulated_paths
        time_grid = self.mc_engine.time_grid
        
        # Use mean reversion level from data if not provided
        if mu is None:
            mu = self.commodity_data.mean_reversion_level
        
        # Select random paths to plot
        n_total = paths.shape[0]
        indices = np.random.choice(n_total, size=min(n_paths, n_total), replace=False)
        selected_paths = paths[indices, :]
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual paths with transparency
        for i in range(selected_paths.shape[0]):
            ax.plot(
                time_grid,
                selected_paths[i, :],
                alpha=0.3,
                linewidth=1,
                color='steelblue'
            )
        
        # Plot mean reversion level
        ax.axhline(
            y=mu,
            color='red',
            linestyle='--',
            linewidth=2.5,
            label=f'Long-Term Mean (μ = ${mu:.2f})',
            zorder=10
        )
        
        # Plot initial price
        S0 = paths[0, 0]
        ax.axhline(
            y=S0,
            color='green',
            linestyle=':',
            linewidth=2,
            label=f'Initial Price (S₀ = ${S0:.2f})',
            zorder=10
        )
        
        # Calculate and plot mean path
        mean_path = np.mean(paths, axis=0)
        ax.plot(
            time_grid,
            mean_path,
            color='darkblue',
            linewidth=3,
            label='Mean Path',
            zorder=11
        )
        
        # Formatting
        ax.set_xlabel('Time (Years)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title(
            f'Monte Carlo Simulation: {selected_paths.shape[0]} Price Paths\n'
            f'Ornstein-Uhlenbeck Mean-Reverting Process',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_payoff_distribution(
        self,
        strike: float,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Visualize the distribution of option payoffs.
        
        Args:
            strike: Strike price
            figsize: Figure size
        """
        if self.option_pricer.payoffs is None:
            raise ValueError("No payoffs calculated. Price option first.")
        
        payoffs = self.option_pricer.payoffs
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left panel: Histogram of all payoffs
        ax1.hist(
            payoffs,
            bins=50,
            color='steelblue',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax1.axvline(
            x=np.mean(payoffs),
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean Payoff: ${np.mean(payoffs):.4f}'
        )
        ax1.axvline(
            x=self.option_pricer.option_price / np.exp(-0.042 * 1),  # Undiscounted
            color='green',
            linestyle='--',
            linewidth=2,
            label=f'Expected Payoff: ${np.mean(payoffs):.4f}'
        )
        ax1.set_xlabel('Payoff (USD)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Option Payoffs', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Right panel: Histogram of in-the-money payoffs only
        itm_payoffs = payoffs[payoffs > 0]
        ax2.hist(
            itm_payoffs,
            bins=30,
            color='green',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        ax2.axvline(
            x=np.mean(itm_payoffs) if len(itm_payoffs) > 0 else 0,
            color='darkgreen',
            linestyle='--',
            linewidth=2,
            label=f'Mean ITM Payoff: ${np.mean(itm_payoffs) if len(itm_payoffs) > 0 else 0:.4f}'
        )
        ax2.set_xlabel('Payoff (USD)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title(
            f'In-The-Money Payoffs Only\n'
            f'({100 * len(itm_payoffs) / len(payoffs):.1f}% of simulations)',
            fontsize=12,
            fontweight='bold'
        )
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_report(
        self,
        strike: float,
        delta: float,
        save_path: Optional[str] = None
    ):
        """
        Generate a comprehensive risk report with all visualizations.
        
        Args:
            strike: Strike price
            delta: Calculated delta
            save_path: Optional path to save the figure
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Panel 1: Historical prices (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        data = self.commodity_data.data
        ax1.plot(data.index, data['Close'], color='steelblue', linewidth=1.5)
        ax1.axhline(
            y=self.commodity_data.mean_reversion_level,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Mean: ${self.commodity_data.mean_reversion_level:.2f}'
        )
        ax1.set_title('Historical Price Data', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=10)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Simulated paths (top right)
        ax2 = fig.add_subplot(gs[0, 1])
        paths = self.mc_engine.simulated_paths
        time_grid = self.mc_engine.time_grid
        n_display = 30
        indices = np.random.choice(paths.shape[0], size=n_display, replace=False)
        for i in indices:
            ax2.plot(time_grid, paths[i, :], alpha=0.3, linewidth=1, color='steelblue')
        ax2.axhline(
            y=self.commodity_data.mean_reversion_level,
            color='red',
            linestyle='--',
            linewidth=2
        )
        ax2.plot(time_grid, np.mean(paths, axis=0), color='darkblue', linewidth=2.5, label='Mean Path')
        ax2.set_title(f'Simulated Price Paths (n={n_display})', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Price (USD)', fontsize=10)
        ax2.set_xlabel('Time (Years)', fontsize=10)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Terminal price distribution (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        terminal_prices = self.mc_engine.get_terminal_prices()
        ax3.hist(terminal_prices, bins=50, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.axvline(x=np.mean(terminal_prices), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: ${np.mean(terminal_prices):.2f}')
        ax3.set_title('Terminal Price Distribution', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Price at Maturity (USD)', fontsize=10)
        ax3.set_ylabel('Frequency', fontsize=10)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Average price distribution (middle right)
        ax4 = fig.add_subplot(gs[1, 1])
        avg_prices = np.mean(paths, axis=1)
        ax4.hist(avg_prices, bins=50, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axvline(x=strike, color='red', linestyle='--', linewidth=2, label=f'Strike: ${strike:.2f}')
        ax4.axvline(x=np.mean(avg_prices), color='darkgreen', linestyle='--', linewidth=2,
                   label=f'Mean: ${np.mean(avg_prices):.2f}')
        ax4.set_title('Average Price Distribution (Asian)', fontsize=11, fontweight='bold')
        ax4.set_xlabel('Average Price (USD)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Payoff distribution (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        payoffs = self.option_pricer.payoffs
        ax5.hist(payoffs, bins=50, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax5.axvline(x=np.mean(payoffs), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: ${np.mean(payoffs):.4f}')
        ax5.set_title('Option Payoff Distribution', fontsize=11, fontweight='bold')
        ax5.set_xlabel('Payoff (USD)', fontsize=10)
        ax5.set_ylabel('Frequency', fontsize=10)
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Summary statistics (bottom right)
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        
        summary_text = f"""
QUANTITATIVE PRICING SUMMARY
{'='*45}

MARKET PARAMETERS:
  Spot Price (S₀):           ${paths[0, 0]:.2f}
  Mean Reversion Level (μ):  ${self.commodity_data.mean_reversion_level:.2f}
  Volatility (σ):            {self.commodity_data.annualized_volatility*100:.2f}%
  
SIMULATION RESULTS:
  Number of Paths:           {paths.shape[0]:,}
  Time Steps:                {paths.shape[1]-1:,}
  Mean Terminal Price:       ${terminal_prices.mean():.2f}
  Std Dev Terminal:          ${terminal_prices.std():.2f}
  
OPTION PRICING:
  Option Type:               Asian Call
  Strike Price (K):          ${strike:.2f}
  Risk-Free Rate:            4.20%
  Time to Maturity:          1.00 year
  
  OPTION PRICE:              ${self.option_pricer.option_price:.4f}
  
  Expected Payoff:           ${np.mean(payoffs):.4f}
  Payoff Std Dev:            ${np.std(payoffs):.4f}
  Prob. In-The-Money:        {100*np.sum(payoffs>0)/len(payoffs):.2f}%
  
GREEKS:
  Delta (Δ):                 {delta:.4f}
  Interpretation:            ${1:.0f} ↑ → ${delta:.4f} ↑
        """
        
        ax6.text(0.1, 0.95, summary_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        fig.suptitle(
            'COMPREHENSIVE COMMODITY DERIVATIVES RISK REPORT',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nComprehensive report saved to: {save_path}")
        
        return fig


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def main():
    """
    Main execution function that integrates all modules and runs the complete
    quantitative commodity pricing analysis.
    """
    print("\n" + "="*70)
    print("QUANTITATIVE COMMODITY PRICING ENGINE")
    print("Integrated Analysis: Data → Simulation → Pricing → Risk")
    print("="*70 + "\n")
    
    # ========================================================================
    # MODULE 1: DATA INGESTION & PARAMETER ESTIMATION
    # ========================================================================
    print("STEP 1: Downloading and analyzing historical commodity data...")
    print("-" * 70)
    
    commodity = CommodityData(ticker='CL=F', years=5)
    params = commodity.analyze(use_simulation=False)  # Try real data first
    commodity.print_summary()
    
    # Extract parameters for simulation
    S0 = params['current_price']
    mu = params['mean_reversion_level']
    sigma = params['annualized_volatility']
    
    # ========================================================================
    # MODULE 2: MONTE CARLO SIMULATION (ORNSTEIN-UHLENBECK)
    # ========================================================================
    print("\nSTEP 2: Running Monte Carlo simulation with OU process...")
    print("-" * 70)
    
    # Simulation parameters
    theta = 0.15          # Mean reversion speed (calibrated for commodities)
    T = 1.0              # 1 year time horizon
    n_steps = 252        # Daily time steps (252 trading days)
    n_simulations = 10000  # 10,000 price paths
    
    # Initialize and run simulation
    mc_engine = MonteCarloEngine(random_seed=42)
    simulated_paths = mc_engine.simulate_ou_process(
        S0=S0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations
    )
    
    mc_engine.print_simulation_summary()
    
    # ========================================================================
    # MODULE 3: OPTION PRICING (ASIAN CALL)
    # ========================================================================
    print("\nSTEP 3: Pricing Asian Call Option...")
    print("-" * 70)
    
    # Option parameters
    strike = S0  # At-the-money (ATM) option
    risk_free_rate = 0.042  # 4.2% annual risk-free rate
    
    # Price the option
    pricer = OptionPricer(mc_engine)
    option_price = pricer.price_asian_call(
        strike=strike,
        risk_free_rate=risk_free_rate,
        T=T
    )
    
    print(f"\nCalculating Greeks (Delta)...")
    print("-" * 70)
    
    # Calculate Delta
    delta = pricer.calculate_delta(
        S0=S0,
        mu=mu,
        theta=theta,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_simulations=n_simulations,
        strike=strike,
        risk_free_rate=risk_free_rate,
        bump_size=1.0
    )
    
    pricer.print_pricing_summary(strike=strike, delta=delta)
    
    # ========================================================================
    # MODULE 4: RISK ANALYSIS & VISUALIZATION
    # ========================================================================
    print("\nSTEP 4: Generating risk analysis and visualizations...")
    print("-" * 70)
    
    # Initialize risk analyzer
    risk_analyzer = RiskAnalyzer(commodity, mc_engine, pricer)
    
    # Create individual visualizations
    print("\nGenerating price path visualization...")
    fig1 = risk_analyzer.plot_price_paths(n_paths=50, mu=mu)
    plt.savefig('/home/claude/price_paths.png', dpi=300, bbox_inches='tight')
    
    print("Generating payoff distribution visualization...")
    fig2 = risk_analyzer.plot_payoff_distribution(strike=strike)
    plt.savefig('/home/claude/payoff_distribution.png', dpi=300, bbox_inches='tight')
    
    print("Generating comprehensive risk report...")
    fig3 = risk_analyzer.create_comprehensive_report(
        strike=strike,
        delta=delta,
        save_path='/home/claude/comprehensive_report.png'
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - EXECUTIVE SUMMARY")
    print("="*70)
    print(f"""
Market Parameters:
  Commodity:                 WTI Crude Oil Futures (CL=F)
  Current Spot Price:        ${S0:.2f}
  Long-Term Mean:            ${mu:.2f}
  Annualized Volatility:     {sigma*100:.2f}%
  Mean Reversion Speed:      {theta:.2f}

Option Contract:
  Type:                      Asian Call (Arithmetic Average)
  Strike Price:              ${strike:.2f} (ATM)
  Time to Maturity:          {T:.1f} year
  Risk-Free Rate:            {risk_free_rate*100:.1f}%

Pricing Results:
  OPTION PRICE:              ${option_price:.4f} per barrel
  Expected Payoff:           ${np.mean(pricer.payoffs):.4f}
  Probability ITM:           {100*np.sum(pricer.payoffs>0)/len(pricer.payoffs):.2f}%
  Delta:                     {delta:.4f}

Business Interpretation:
  For a quarterly contract of 100,000 barrels:
  - Total Premium:           ${option_price * 100000:,.2f}
  - Hedge Ratio (Delta):     {delta:.2%}
  - Risk Reduction:          Locks in average price ≤ ${strike:.2f}/barrel
    """)
    
    print("\nAll visualizations saved successfully!")
    print("Files generated:")
    print("  - price_paths.png")
    print("  - payoff_distribution.png")
    print("  - comprehensive_report.png")
    print("="*70 + "\n")
    
    return commodity, mc_engine, pricer, risk_analyzer


if __name__ == "__main__":
    # Execute the complete analysis
    commodity_obj, mc_obj, pricer_obj, risk_obj = main()
    
    # Show plots
    plt.show()
