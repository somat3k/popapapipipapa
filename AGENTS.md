# AGENTS & TODO — Multiplex Financials Platform

> **Purpose**: Comprehensive agent registry, system TODO list, and Epics A–Z with
> sub-goals and implementation tasks for the Multiplex Financials DEFI AI Platform.

---

## System Overview

Multiplex Financials is a DeFi-centric AI platform built around:

- **Autonomous AI Agents** with tool-use capabilities and agentic chat
- **ML Supervised Learning Modules** with dynamic hyperparameter adjustment
- **Morpho Protocol Integration** on Polygon for collateral swap, supply, borrow,
  and repay mechanics
- **Adaptive Trading Chain** with algorithm-driven consistency and metrics evaluation
- **Sophisticated Python GUI** (tkinter) with windowed frames and live dashboards

---

## Agent Registry

| ID  | Agent Name          | Role                                        | Status     |
|-----|---------------------|---------------------------------------------|------------|
| A01 | OrchestratorAgent   | Master coordinator; routes tasks to peers   | Active     |
| A02 | TradingAgent        | Executes trading algorithms and strategies  | Active     |
| A03 | DeFiAgent           | Handles Morpho supply/borrow/repay/swap     | Active     |
| A04 | MLAgent             | Trains and evaluates supervised ML models   | Active     |
| A05 | AnalysisAgent       | Market data analysis and signal generation  | Active     |
| A06 | ChatAgent           | Agentic chat interface with tool dispatch   | Active     |
| A07 | RiskAgent           | Portfolio risk scoring and circuit-breakers | Active     |
| A08 | DataFeedAgent       | Fetches and normalises on-chain / off-chain  | Active     |
| A09 | HyperparamAgent     | Dynamic hyperparameter tuning and BO search | Active     |
| A10 | MetricsAgent        | Collects, aggregates, and exports metrics   | Active     |

---

## Epics A–Z

Each epic contains at minimum 18 granular tasks.

---

### Epic A — Agent Framework Foundation

**Goal**: Build a robust, extensible base for all AI agents in the system.

1. Define `BaseAgent` abstract class with `run()`, `stop()`, `reset()` lifecycle methods
2. Implement agent state machine (`IDLE`, `RUNNING`, `PAUSED`, `ERROR`, `STOPPED`)
3. Create `AgentRegistry` singleton for agent discovery and lookup by ID
4. Build `MessageBus` pub/sub for inter-agent communication
5. Add `ToolRegistry` with dynamic tool registration and schema validation
6. Implement `AgentContext` object for passing shared state between agents
7. Support agent chaining via `pipeline()` combinator
8. Add coroutine / asyncio event-loop support per agent
9. Implement agent health-checks with configurable heartbeat interval
10. Build serialisation layer (JSON/protobuf) for agent messages
11. Add logging adapter that tags every log entry with agent ID
12. Implement retry logic with exponential back-off for failed tool calls
13. Support capability negotiation between agents (skills manifest)
14. Create unit tests for every lifecycle transition
15. Add `__repr__` / `__str__` for all agent objects for debug output
16. Document agent API surface in docstrings (Google style)
17. Provide example custom agent template (`TemplateAgent`)
18. Add CLI command `agents list` to display running agents and their status
19. Implement graceful shutdown sequencing (drain queues before stopping)
20. Instrument agent event latency using `time.perf_counter`

---

### Epic B — Base Tools & Utilities

**Goal**: Provide reusable tool primitives consumed by all agents.

1. `WebFetchTool` — HTTP GET/POST with retry, timeout, and response caching
2. `PriceFeedTool` — Fetch token prices via CoinGecko / CoinMarketCap APIs
3. `CalculatorTool` — Safe arithmetic evaluation with unit awareness
4. `TimestampTool` — Timezone-aware datetime helpers and unix epoch converters
5. `JSONParserTool` — Schema-validated JSON parsing with error recovery
6. `CSVReaderTool` — Streaming CSV ingestion for large financial datasets
7. `DataFrameTool` — Pandas DataFrame wrappers with type enforcement
8. `StatsTool` — Descriptive statistics (mean, std, skew, kurtosis, VaR)
9. `CryptographyTool` — Hash, HMAC, and key-derivation helpers
10. `ConfigLoaderTool` — YAML/TOML/ENV config loading with defaults
11. `AlertTool` — Send notifications (email, Telegram, Slack webhook)
12. `LogAggregatorTool` — Aggregate structured logs from multiple agents
13. `CacheTool` — TTL-based in-memory cache with LRU eviction
14. `SchedulerTool` — Cron-style task scheduler for periodic agent actions
15. `CircuitBreakerTool` — Trip after N failures; auto-reset after timeout
16. `VectorStoreTool` — Simple embedding store for semantic retrieval
17. `SummaryTool` — LLM-backed summarisation of financial reports
18. `UniqueIDTool` — Thread-safe unique ID generation (UUID4 / snowflake)
19. `ValidationTool` — Pydantic-style runtime schema enforcement
20. `AuditTool` — Immutable append-only audit log with tamper detection

---

### Epic C — Chat & Agentic Interface

**Goal**: Build the agentic chat subsystem for human-agent interaction.

1. Implement `ChatSession` class with conversation history management
2. Build prompt-template system with variable interpolation
3. Integrate tool-use loop: parse LLM output → dispatch tool → inject result
4. Support multi-turn dialogue with context windowing (token budget)
5. Add streaming token output to chat panel via generator / callback
6. Implement intent classifier to route queries to specialist agents
7. Build slash-command parser (`/defi`, `/trade`, `/ml`, `/help`)
8. Add response formatting layer (markdown → styled tkinter text)
9. Implement conversation export to JSON and PDF
10. Support conversation branching and rewind
11. Add sentiment analysis on user messages to adjust agent tone
12. Build persona system allowing agents to adopt financial personas
13. Implement rate-limiting for chat queries to prevent abuse
14. Add typing indicator / spinner in GUI during agent processing
15. Implement auto-save of conversation history to SQLite
16. Add `@mention` syntax to route message to specific agent
17. Build FAQ retrieval from embedded knowledge base
18. Support image/chart attachment in chat (display matplotlib figures inline)
19. Add voice input hook (placeholder for speech-to-text integration)
20. Write integration tests covering multi-turn tool-use scenarios

---

### Epic D — DeFi Protocol Core

**Goal**: Implement DeFi protocol abstractions for Polygon interactions.

1. Define `DeFiProtocol` abstract base class with `supply()`, `borrow()`, `repay()`, `withdraw()`, `getPosition()` methods
2. Implement Web3 provider management with fallback RPC endpoints
3. Add ABI registry for all smart contract interfaces
4. Build transaction builder with EIP-1559 gas estimation
5. Implement nonce manager with optimistic concurrency control
6. Add transaction receipt polling with configurable confirmation threshold
7. Implement `approve()` helper for ERC-20 token allowances
8. Build multi-call batching to reduce RPC round-trips
9. Add slippage protection layer for swap transactions
10. Implement deadline-aware transaction submission
11. Build position tracker updating balances after each transaction
12. Add event listener for `Supply`, `Borrow`, `Repay`, `Withdraw` events
13. Implement health-factor monitor with configurable alert thresholds
14. Add gas price oracle with EIP-1559 fee suggestion
15. Build transaction simulation via `eth_call` before broadcast
16. Implement wallet abstraction supporting EOA and smart-contract wallets
17. Add network switching between Polygon mainnet and Mumbai testnet
18. Write mock provider for unit tests (no real RPC needed)
19. Add retry logic for reverted transactions with adjusted gas
20. Document all contract addresses and ABI hashes in `config/contracts.yaml`

---

### Epic E — Morpho Protocol Integration

**Goal**: Full integration with Morpho Blue on Polygon.

1. Implement `MorphoClient` wrapping Morpho Blue contract calls
2. Support `supply(market, amount, onBehalf)` with max-approval helper
3. Support `borrow(market, amount, onBehalf, receiver)` with health-check guard
4. Support `repay(market, amount, onBehalf, data)` including partial repay
5. Support `withdraw(market, amount, onBehalf, receiver)` with share accounting
6. Implement `collateralSwap()` reducing borrowed amount via flash-loan-like mechanic
7. Add `getMarketParams(id)` to fetch market configuration from on-chain
8. Build `getPosition(market, user)` returning supply/borrow shares and collateral
9. Implement health-factor calculation using Morpho's liquidation LTV
10. Add liquidation risk monitor with auto-repay trigger
11. Implement supply APY and borrow APY calculation from on-chain accruedInterest
12. Build market selector scoring markets by liquidity, APY, and risk
13. Add slippage guard for collateral swap to reject poor-rate swaps
14. Implement `increaseCollateral()` helper for deleveraging
15. Support Oracle price feed integration (Chainlink aggregators)
16. Add event indexing for Morpho events to reconstruct history
17. Implement position snapshot and state diff for audit
18. Build CLI command `defi status` showing current positions
19. Add alerting when borrowed utilisation exceeds 80%
20. Write comprehensive tests against Hardhat fork of Polygon mainnet

---

### Epic F — Collateral Swap Engine

**Goal**: Implement the collateral swap flow to reduce borrowed amount.

1. Design `CollateralSwapEngine` class with input/output asset params
2. Implement step 1: calculate required swap amount to achieve target LTV
3. Implement step 2: fetch best swap route (1inch / Paraswap aggregator)
4. Implement step 3: execute swap via DEX aggregator
5. Implement step 4: supply swapped tokens as additional collateral on Morpho
6. Implement step 5: repay portion of borrow using freed collateral value
7. Add atomicity guard — revert entire flow if any step fails
8. Implement dry-run mode returning projected outcome without broadcasting
9. Build `CollateralSwapSimulator` for what-if analysis
10. Add slippage and price-impact validation before execution
11. Implement gas cost estimation for entire multi-step flow
12. Support partial collateral swap (percentage-based target)
13. Add before/after position comparison report
14. Implement re-entrancy protection checks
15. Build monitoring hook that triggers swap when LTV exceeds threshold
16. Add approval management for both input and output tokens
17. Implement multi-hop swap path optimisation
18. Support WMATIC / USDC / WBTC / WETH as collateral assets
19. Write test scenarios: healthy position, near-liquidation, and over-collateralised
20. Document flow diagram in docstring with ASCII art

---

### Epic G — Trading Algorithm Framework

**Goal**: Build pluggable trading algorithm infrastructure.

1. Define `Algorithm` abstract base with `generate_signals()`, `on_bar()`, `on_tick()` hooks
2. Implement `MeanReversionAlgo` using Bollinger Bands
3. Implement `MomentumAlgo` using RSI and MACD
4. Implement `TrendFollowingAlgo` using EMA crossover
5. Implement `ArbitrageAlgo` for cross-exchange price discrepancies
6. Implement `MarketMakingAlgo` with bid/ask spread management
7. Add `SignalAggregator` combining multiple algorithm outputs with weights
8. Build `Backtester` for historical strategy evaluation
9. Implement `Portfolio` object tracking positions and PnL
10. Add position sizing via Kelly criterion and fixed fractional methods
11. Implement stop-loss and take-profit order management
12. Add transaction cost model (spread + commission + slippage)
13. Build `OrderBook` simulator for backtesting
14. Implement walk-forward optimisation for strategy parameters
15. Add regime detection (trending vs. ranging market) to switch algorithms
16. Build performance report generator (Sharpe, Sortino, max drawdown)
17. Add Monte Carlo simulation for strategy robustness
18. Implement live trading mode connecting to DEX (Uniswap V3 on Polygon)
19. Write backtests for all algorithms on historical MATIC/USDC data
20. Add real-time chart rendering of signals in trading panel

---

### Epic H — Trading Chain & Consistency

**Goal**: Ensure consistent, ordered execution across the trading pipeline.

1. Design `TradingChain` as an ordered list of `ChainStep` processors
2. Implement `DataIngestionStep` — fetches and validates OHLCV data
3. Implement `PreprocessingStep` — normalises, fills gaps, computes features
4. Implement `FeatureEngineeringStep` — rolling stats, lagged features, TA indicators
5. Implement `ModelInferenceStep` — runs trained ML model to generate predictions
6. Implement `SignalGenerationStep` — converts predictions to buy/sell signals
7. Implement `RiskFilterStep` — filters signals that violate risk limits
8. Implement `OrderGenerationStep` — sizes and places orders
9. Implement `ExecutionStep` — sends orders to exchange / DEX
10. Implement `ConfirmationStep` — polls for order fill confirmation
11. Implement `AccountingStep` — updates portfolio state and PnL
12. Add chain-level error handling with fallback steps
13. Implement `ChainContext` shared state object passed through all steps
14. Add per-step timing and profiling instrumentation
15. Build chain replay from saved context snapshots
16. Implement idempotency keys to prevent duplicate order submission
17. Add dry-run / paper-trading mode for chain without order execution
18. Write integration test running full chain on synthetic data
19. Implement chain configuration via YAML pipeline file
20. Add Prometheus metrics export from chain step execution

---

### Epic I — ML Data Pipeline

**Goal**: Automated data ingestion, feature engineering, and dataset preparation for ML.

1. Build `DataPipeline` class orchestrating raw → processed data flow
2. Implement OHLCV loader from CSV, Parquet, and REST API
3. Add on-chain data loader (Dune Analytics / The Graph queries)
4. Implement `FeatureStore` with versioned feature sets
5. Build technical indicator library (RSI, MACD, ATR, OBV, Stochastic)
6. Add DeFi-specific features: TVL, borrow utilisation, liquidation volume
7. Implement outlier detection and removal (IQR, Z-score methods)
8. Add missing value imputation strategies (forward-fill, interpolation, median)
9. Build train/validation/test temporal split without data leakage
10. Implement feature normalisation (min-max, z-score, robust scaler)
11. Add feature importance ranking using permutation importance
12. Implement correlation analysis and collinear feature removal
13. Build label generation for classification (direction) and regression (return)
14. Add walk-forward cross-validation splitter
15. Implement data augmentation (jitter, time-warp) for small datasets
16. Build dataset versioning with hash-based content addressing
17. Add data drift detection using KL divergence and PSI
18. Implement streaming data processor for real-time feature computation
19. Write unit tests for each pipeline stage with synthetic data
20. Export pipeline config and feature schema to YAML

---

### Epic J — Supervised ML Models

**Goal**: Implement supervised learning models for price direction and return prediction.

1. Implement `BaseModel` abstract class with `fit()`, `predict()`, `evaluate()` methods
2. Implement `LinearRegressionModel` with regularisation (Ridge/Lasso)
3. Implement `RandomForestModel` with feature importance output
4. Implement `GradientBoostingModel` (XGBoost / LightGBM wrapper)
5. Implement `NeuralNetworkModel` using scikit-learn for sequence-friendly data
6. Implement `TransformerModel` (lightweight time-series transformer)
7. Implement `EnsembleModel` averaging predictions from multiple base models
8. Add `ModelRegistry` for versioned model storage and retrieval
9. Implement model serialisation/deserialisation (pickle + ONNX export)
10. Build `ModelValidator` computing out-of-sample metrics before promotion
11. Add `ModelMonitor` detecting prediction drift in production
12. Implement calibration module for probability outputs
13. Add SHAP explainability integration for feature attribution
14. Build A/B testing framework for comparing model versions
15. Implement online learning hook for incremental model updates
16. Add confidence interval estimation for regression predictions
17. Implement multi-output model supporting simultaneous prediction of multiple targets
18. Write comprehensive test suite with synthetic datasets
19. Add model card generator documenting architecture, metrics, and limitations
20. Implement `predict_batch()` with vectorised inference for throughput

---

### Epic K — Dynamic Hyperparameter Adjustment

**Goal**: Automate and adapt ML hyperparameters during training.

1. Define `HyperparamSpace` dataclass with min, max, step, and type per param
2. Implement grid search over discrete param space
3. Implement random search with configurable budget
4. Implement Bayesian optimisation using `scikit-optimize` / `optuna`
5. Add `HyperparamScheduler` adjusting LR, dropout, and batch size on-the-fly
6. Implement learning rate warm-up and cosine annealing schedule
7. Add early stopping with patience and delta threshold
8. Implement adaptive batch sizing based on GPU memory utilisation
9. Build `HyperparamLogger` recording every trial with params and metrics
10. Add population-based training (PBT) for continuous adaptation
11. Implement curriculum learning schedule (easy → hard samples)
12. Add gradient clipping with dynamic threshold based on gradient norm history
13. Implement weight decay schedule tied to validation loss plateau
14. Build `HyperparamDashboard` displaying training dynamics in GUI
15. Add multi-objective optimisation (accuracy vs. inference latency)
16. Implement transfer-learning warmstart loading pre-trained weights
17. Add automated feature selection as part of hyperparam search
18. Build hyperparameter importance ranking via fANOVA
19. Implement pruning of unpromising trials (Hyperband / ASHA)
20. Write tests verifying that scheduler produces monotonically decreasing LR

---

### Epic L — Model Training Pipeline

**Goal**: End-to-end supervised training with monitoring and reproducibility.

1. Implement `Trainer` class orchestrating data loading → training → evaluation
2. Add reproducibility: seed everything (random, numpy, sklearn)
3. Implement gradient accumulation for large effective batch sizes
4. Build checkpointing: save best model by validation metric
5. Implement mixed-precision training (FP16 / BF16 where supported)
6. Add training loss and validation metric logging per epoch
7. Implement multi-GPU / multi-process training support (DataParallel)
8. Build live training curve visualisation in GUI (matplotlib canvas)
9. Add `TrainingCallback` interface for extensible hooks
10. Implement `EarlyStoppingCallback` tied to validation metric
11. Implement `LRSchedulerCallback` wrapping dynamic LR adjustment
12. Add gradient norm tracking per layer for debugging
13. Implement dataset shuffling with temporal integrity preservation
14. Build experiment tracker logging to CSV / W&B-compatible format
15. Add automatic hyperparameter tuning run before full training
16. Implement training resumption from checkpoint on crash
17. Add training time estimation based on epoch 1 duration
18. Build final model evaluation report: metrics, confusion matrix, feature importance
19. Implement K-fold ensemble training and averaging
20. Add model serving benchmark measuring latency and throughput post-training

---

### Epic M — Metrics & Evaluation Framework

**Goal**: Comprehensive financial and ML metrics for model and strategy evaluation.

1. Implement `MetricsEngine` collecting real-time trading metrics
2. Add `SharpeRatio` computation with annualisation factor
3. Add `SortinoRatio` using downside deviation
4. Add `MaxDrawdown` and `CalmarRatio` computation
5. Add `WinRate`, `ProfitFactor`, and `ExpectedValue` for strategy analysis
6. Implement `PrecisionRecall` for ML classification evaluation
7. Add `RMSE`, `MAE`, `MAPE` for regression evaluation
8. Implement `DirectionalAccuracy` (% correct direction predictions)
9. Add `InformationCoefficient` (rank correlation of predictions with returns)
10. Build `MetricsDashboard` widget in GUI showing live metric tiles
11. Implement metric time-series storage for trend analysis
12. Add metric export to Prometheus / Grafana-compatible format
13. Build periodic metric report generator (daily/weekly summary)
14. Implement benchmark comparison against buy-and-hold baseline
15. Add statistical significance testing (t-test) between strategy variants
16. Implement rolling metric windows (7d, 30d, 90d)
17. Add risk-adjusted return decomposition (alpha, beta, factor exposures)
18. Build metric alert system triggering on threshold breaches
19. Implement metric correlation matrix visualisation
20. Write tests verifying all metric implementations against known values

---

### Epic N — Risk Management

**Goal**: Comprehensive risk controls protecting capital and DeFi positions.

1. Implement `RiskEngine` evaluating portfolio-level risk in real time
2. Add position-size limit per asset (% of portfolio NAV)
3. Implement Value-at-Risk (VaR) with historical simulation and parametric methods
4. Add Conditional VaR (CVaR / Expected Shortfall)
5. Implement correlation-aware portfolio risk (covariance matrix)
6. Add DeFi-specific health-factor monitoring for Morpho positions
7. Implement liquidation price calculator per collateral position
8. Add circuit-breaker halting trading on drawdown threshold breach
9. Implement maximum leverage limiter across DeFi positions
10. Build `RiskReport` generating daily risk summary
11. Add volatility scaling for position sizing in high-vol regimes
12. Implement counterparty risk scoring for DeFi protocols
13. Add tail-risk hedging recommendations based on stress tests
14. Implement stress-test scenarios: -50% asset price, gas spike, liquidity crisis
15. Build risk attribution report decomposing portfolio risk by asset and strategy
16. Add real-time PnL attribution (delta, gamma, theta components)
17. Implement exposure limit for borrowed amount relative to collateral
18. Build alert system for rapid position change (>X% in Y minutes)
19. Add regulatory compliance checks (placeholder for future requirements)
20. Write simulation tests verifying circuit-breaker triggers correctly

---

### Epic O — On-Chain Data Integration

**Goal**: Real-time and historical on-chain data for analytics and trading.

1. Implement `Web3DataProvider` with multi-chain support
2. Add Polygon RPC provider with automatic failover
3. Build `BlockListener` reacting to new blocks with callback hooks
4. Implement `EventListener` subscribing to specific contract events
5. Add `TransactionTracer` decoding calldata and logs for Morpho txns
6. Build Dune Analytics query client for historical DeFi metrics
7. Implement The Graph subgraph client for Morpho position history
8. Add DEX price oracle reading Uniswap V3 TWAP
9. Implement Chainlink price feed reader for collateral pricing
10. Build `TokenMetrics` aggregating volume, holders, and transfer counts
11. Add TVL tracker for Morpho markets on Polygon
12. Implement APY / APR computation from on-chain accruedInterest
13. Build gas price tracker with EIP-1559 base fee history
14. Add mempool monitoring for pending large transactions
15. Implement wallet analytics: PnL, position history, liquidation events
16. Build data normalisation layer converting raw chain data to DataFrames
17. Add caching layer with block-based invalidation
18. Implement data integrity checks: block hash verification
19. Build backfill mode for historical event ingestion
20. Write mock Web3 provider enabling offline unit tests

---

### Epic P — Portfolio Management

**Goal**: Holistic portfolio tracking across DeFi and trading positions.

1. Implement `Portfolio` class aggregating all asset positions
2. Add `Position` dataclass tracking size, entry price, current price, PnL
3. Implement NAV calculation including DeFi collateral and borrow netted
4. Build `AllocationTarget` for rebalancing (target weights per asset)
5. Implement rebalancing engine computing required trades to reach target
6. Add transaction cost-aware rebalancing (avoid tiny trades)
7. Implement tax-lot accounting (FIFO / LIFO / specific ID)
8. Build portfolio history tracker storing daily snapshots
9. Add performance attribution decomposing returns by asset and strategy
10. Implement `PortfolioDashboard` widget showing allocation pie chart
11. Add benchmark tracking vs. BTC, ETH, and MATIC
12. Implement drawdown tracker with underwater period analysis
13. Build liquidity risk assessment for each position
14. Add correlation matrix visualisation for portfolio assets
15. Implement Markowitz efficient frontier optimisation
16. Add Black-Litterman model for incorporating analyst views
17. Build transaction history export to CSV and PDF
18. Implement reconciliation between local state and on-chain balances
19. Add portfolio clone / copy feature for strategy comparison
20. Write tests verifying NAV calculation with mock price feeds

---

### Epic Q — GUI Main Window & Theming

**Goal**: Build a sophisticated, multi-panel Python tkinter GUI.

1. Implement `MainWindow` with tkinter `Tk` root and 1280×900 minimum size
2. Add dark-mode color theme with finance-appropriate palette
3. Implement tabbed notebook for Dashboard, Trading, DeFi, ML, Chat, Settings
4. Build header bar with platform name, clock, network status, and wallet address
5. Add status bar at bottom showing last action, gas price, and block number
6. Implement resizable panel layout using `PanedWindow`
7. Add left sidebar with agent status indicators (green/red/yellow dots)
8. Build `ThemeManager` supporting light and dark themes with live switching
9. Implement font scaling for HiDPI / Retina displays
10. Add keyboard shortcuts for common actions (`Ctrl+T` trade, `Ctrl+D` DeFi)
11. Implement modal dialog framework for confirmations and config forms
12. Build notification toast system (auto-dismiss after 3 s)
13. Add progress bar widget for long-running operations
14. Implement scrollable log viewer showing agent activity
15. Add chart canvas widget (matplotlib FigureCanvasTkAgg)
16. Implement window state persistence (size, position, tab) across sessions
17. Build about dialog showing version, dependencies, and licence
18. Add onboarding wizard for first-run configuration
19. Implement drag-and-drop panel rearrangement
20. Write smoke tests verifying window launches without errors

---

### Epic R — Dashboard Panel

**Goal**: Live metrics dashboard providing at-a-glance system health.

1. Build `DashboardPanel` with grid of metric tiles
2. Add `MetricTile` widget showing label, value, and delta indicator
3. Implement PnL tile with colour-coded up/down
4. Add portfolio NAV tile with sparkline
5. Implement DeFi health-factor tile with progress bar
6. Add active agent count tile
7. Implement active positions tile listing top 5 positions by size
8. Add ML model accuracy tile showing last evaluation score
9. Implement gas price tile with historical 24h chart
10. Add market overview: MATIC, ETH, BTC prices with 24h change
11. Implement alert ticker at top of dashboard scrolling recent alerts
12. Build auto-refresh loop updating all tiles every 5 seconds
13. Add collapsible section for DeFi position details
14. Implement chart toggle showing 1h/4h/1d/1w timeframes
15. Add right-click context menu on tiles for drill-down
16. Build export dashboard snapshot to PNG button
17. Implement dark/light mode switch on dashboard
18. Add drag-reorder for metric tiles
19. Implement compact vs. expanded view toggle
20. Write tests verifying tile values update correctly on data change

---

### Epic S — Trading Panel

**Goal**: Interface for creating, monitoring, and managing trades.

1. Build `TradingPanel` with strategy selector and parameter inputs
2. Add pair selector widget (MATIC/USDC, ETH/USDC, BTC/USDC)
3. Implement order form: type (market/limit), side, amount, price
4. Add trade preview showing estimated cost, fees, and slippage
5. Implement live OHLCV candlestick chart (custom canvas or mplfinance)
6. Add signal overlay on chart from selected algorithm
7. Implement order book display (simplified bid/ask ladder)
8. Build active orders list with cancel button
9. Add trade history table with sortable columns
10. Implement strategy backtesting trigger and results display
11. Add algorithm parameters form dynamically generated from schema
12. Implement risk check summary before order submission
13. Build PnL tracker showing open and realised PnL
14. Add one-click DCA (dollar-cost averaging) setup form
15. Implement trade size calculator based on risk % of portfolio
16. Add chart drawing tools (horizontal line, trend line)
17. Implement alert creation on chart price level
18. Build CSV export for trade history
19. Add confirmation dialog before order submission
20. Write GUI integration tests for order form validation

---

### Epic T — DeFi Panel

**Goal**: Full-featured panel for Morpho DeFi operations.

1. Build `DeFiPanel` with Morpho market selector
2. Add supply form: token, amount, and APY preview
3. Implement borrow form: token, amount, health-factor impact preview
4. Add repay form: partial or full repay with outstanding balance display
5. Implement withdraw form with available liquidity check
6. Build collateral swap wizard: multi-step form with dry-run preview
7. Add position summary card: collateral, debt, health-factor, LTV
8. Implement market table showing all available Morpho markets
9. Add APY comparison bar chart for supply and borrow rates
10. Implement liquidation price calculator widget
11. Build transaction history table for all DeFi operations
12. Add gas estimator for each operation type
13. Implement approval management widget
14. Add auto-repay configuration form (trigger health-factor)
15. Build portfolio health dashboard showing risk across all positions
16. Implement market depth visualisation (supply/borrow distribution)
17. Add PnL tracker for DeFi yield earned vs. interest paid
18. Implement multi-market position aggregation view
19. Build one-click deleveraging button
20. Write GUI tests for supply/borrow form validation

---

### Epic U — ML Panel

**Goal**: GUI panel for model training, evaluation, and management.

1. Build `MLPanel` with model selector dropdown
2. Add dataset loader widget: file picker and data preview table
3. Implement training configuration form: model type, hyperparams, epochs
4. Add hyperparameter search configuration widget
5. Implement training progress bar and live loss curve chart
6. Add validation metrics table updating per epoch
7. Build model registry table showing all saved models
8. Add model comparison view: side-by-side metric tables
9. Implement feature importance bar chart
10. Add confusion matrix heatmap (for classifiers)
11. Build residual plot (for regressors)
12. Implement SHAP waterfall plot for individual prediction explanation
13. Add model deployment toggle (promote model to production)
14. Build hyperparameter tuning results scatter plot
15. Implement dataset statistics panel (distribution histograms)
16. Add data drift report panel
17. Build training log viewer (scrollable text area)
18. Implement model export button (pickle / ONNX)
19. Add learning rate schedule visualiser
20. Write GUI tests verifying panel populates correctly after training

---

### Epic V — Wallet & Account Management

**Goal**: Secure wallet integration and account configuration.

1. Implement `WalletManager` abstracting key storage
2. Support read-only mode using just wallet address (no private key)
3. Add encrypted private key storage with passphrase (AES-256-GCM)
4. Implement WalletConnect integration (placeholder hook)
5. Build `AccountDashboard` showing balances for all tokens
6. Add token approval manager listing and revoking existing approvals
7. Implement multi-wallet support with named profiles
8. Build address book for frequently used addresses
9. Add wallet import via mnemonic (BIP-39) with derivation path support
10. Implement hardware wallet stub (Ledger integration placeholder)
11. Build token balance refresh on new block
12. Add transaction history fetcher from Polygonscan API
13. Implement wallet PnL tracker (unrealised + realised)
14. Build gas tank refill reminder when MATIC balance is low
15. Add wallet backup export to encrypted JSON
16. Implement multi-sig wallet stub for team use
17. Build account settings form (network, default slippage, gas preset)
18. Add session timeout and auto-lock mechanism
19. Implement wallet health check (balance thresholds for operations)
20. Write tests for encrypted key storage and retrieval

---

### Epic W — Configuration & Settings

**Goal**: Centralised, validated configuration management for the entire platform.

1. Implement `Config` dataclass hierarchy with nested sections
2. Add YAML configuration loader with schema validation
3. Implement environment variable overrides for all config values
4. Build `SettingsPanel` GUI form auto-generated from config schema
5. Add section groups: Network, Wallet, Trading, DeFi, ML, Alerts
6. Implement config versioning and migration between versions
7. Add secrets management: API keys stored encrypted in config
8. Build config export / import wizard in GUI
9. Implement profile system: dev, testnet, mainnet profiles
10. Add config validation on startup with clear error messages
11. Implement watch-for-change reload (inotify / polling)
12. Build config documentation generator from dataclass annotations
13. Add sensitive field masking in logs and UI
14. Implement config reset to defaults button in settings panel
15. Add per-agent configuration overrides
16. Build network topology config (RPC endpoints, WSS URLs)
17. Implement feature flags for experimental features
18. Add config audit log tracking who changed what and when
19. Build config comparison tool (current vs. previous version)
20. Write tests verifying validation rejects invalid configurations

---

### Epic X — Logging, Observability & Alerting

**Goal**: Full observability stack for production operation.

1. Implement structured JSON logging with log levels (DEBUG → CRITICAL)
2. Add per-module log filtering via config
3. Implement log rotation with size and time-based policies
4. Build centralised `LogAggregator` collecting from all agents
5. Add Prometheus metrics exporter with `/metrics` endpoint
6. Implement OpenTelemetry traces for end-to-end request tracking
7. Build real-time log viewer in GUI (scrollable, filterable)
8. Add alert rules evaluated against metrics (threshold, rate-of-change)
9. Implement notification dispatch: Telegram, email, webhook
10. Build alert history table in GUI
11. Add on-call schedule integration (placeholder)
12. Implement anomaly detection on key metrics using Z-score
13. Build heartbeat monitor for each agent with dead-man's switch
14. Add slow-query log for RPC calls exceeding 500 ms
15. Implement performance profiling decorator for critical code paths
16. Build dashboard health endpoint returning system status JSON
17. Add disk-space and memory usage monitoring
18. Implement log search/filter widget in GUI
19. Build periodic observability report (hourly summary to log)
20. Write tests verifying log output format matches expected schema

---

### Epic Y — Testing & Quality Assurance

**Goal**: Comprehensive automated test suite ensuring platform reliability.

1. Implement unit test suite using `pytest` for all modules
2. Add `pytest-cov` coverage reporting with 80% minimum threshold
3. Implement integration tests running full trading chain on synthetic data
4. Add contract interaction tests using mock Web3 provider
5. Implement GUI smoke tests launching window and clicking through panels
6. Add property-based tests using `hypothesis` for ML metrics
7. Implement performance benchmarks for critical paths
8. Build regression test suite comparing model metrics to known baselines
9. Add fuzz tests for configuration parsing and user input handling
10. Implement end-to-end test: data ingestion → model training → trade signal
11. Add test fixtures for common objects (mock portfolio, mock market data)
12. Build CI pipeline running tests on push and PR
13. Implement mutation testing to validate test quality
14. Add security scan with `bandit` and `safety`
15. Implement style checking with `flake8` / `black` / `isort`
16. Build API contract tests ensuring agent interfaces are stable
17. Add snapshot tests for GUI widget rendering
18. Implement load tests for metrics endpoint
19. Build chaos test suite injecting RPC failures and validating recovery
20. Write test coverage report to `htmlcov/` and publish as CI artifact

---

### Epic Z — Deployment & Operations

**Goal**: Production-ready deployment, monitoring, and maintenance.

1. Write `Dockerfile` packaging the full application
2. Add `docker-compose.yml` with app, Prometheus, and Grafana services
3. Implement GitHub Actions CI workflow running tests on every push
4. Add semantic versioning with automated changelog generation
5. Build release workflow tagging Docker image on version bump
6. Implement environment-specific deployment configs (dev/staging/prod)
7. Add infrastructure-as-code templates (Terraform) for cloud deployment
8. Implement database migration scripts for SQLite schema changes
9. Build backup and restore procedures for wallet and config data
10. Add monitoring runbook documentation
11. Implement blue-green deployment support
12. Build health-check endpoint used by load balancer / orchestrator
13. Add automatic dependency update bot configuration (Dependabot)
14. Implement log aggregation to centralised store (ELK / Loki)
15. Build performance baseline and regression alerts in CI
16. Add security scanning in CI (SAST, dependency audit)
17. Implement secret rotation procedure documentation
18. Build disaster recovery playbook
19. Add user documentation and installation guide
20. Write architecture decision records (ADRs) for key design choices

---

## Global TODO (Cross-Cutting)

- [ ] Integrate all agents with `AgentRegistry` and `MessageBus`
- [ ] Wire `TradingChain` to live Morpho event stream
- [ ] Complete ML model training pipeline with live data
- [ ] Finish collateral swap engine integration tests on testnet
- [ ] Polish GUI dark theme and responsive layout
- [ ] Add comprehensive docstrings to all public APIs
- [ ] Reach 80%+ test coverage
- [ ] Set up CI/CD pipeline
- [ ] Security audit of private key handling
- [ ] Deploy to Polygon Mumbai testnet and validate end-to-end

---

*Last updated: auto-generated*
