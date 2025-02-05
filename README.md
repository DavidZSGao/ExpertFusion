# ExpertFusion: A Mixture-of-Experts System for Market Prediction

ExpertFusion leverages the powerful Mixture-of-Experts (MoE) architecture to create a dynamic market prediction system. By employing a neural gating network, it intelligently combines predictions from specialized expert models - each focusing on distinct aspects of market analysis (macro, fundamental, news, technical, and risk). The MoE framework enables the system to adaptively weight expert opinions based on market conditions, leading to more robust and context-aware predictions.

## Features

- **Mixture-of-Experts Architecture:**
  - Neural gating network for dynamic expert weighting
  - Specialized expert models with domain-specific analysis
  - Adaptive prediction synthesis based on market context

- **Multiple Expert Models:**
  - MacroExpert: Analyzes GDP, monetary policy, and macro uncertainty
  - FundamentalExpert: Evaluates PE ratios and ROE
  - NewsExpert: Processes news sentiment using GPT-4
  - TechnicalExpert: Analyzes price and volume patterns
  - RiskExpert: Assesses volatility, beta, and liquidity

- **Advanced Data Integration:**
  - Real-time market data via yfinance
  - Fundamental data from WRDS
  - News data from SQLite database
  - Macro uncertainty indicators
  - FOMC statements and policy decisions

- **Machine Learning Components:**
  - Neural network-based gating mechanism
  - GPT-4 powered sentiment analysis
  - Automated feature engineering
  - Portfolio performance analysis

## Project Structure

```
MoE/
├── config.py         # Configuration settings and API keys
├── data_loader.py    # Data loading and preprocessing
├── experts.py        # Expert model implementations
├── main.py          # Main execution script
├── moe_model.py     # MoE model architecture
└── utils.py         # Utility functions and visualization
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- OpenAI API key (for GPT-4 access)
- WRDS account (for fundamental data)
- Required Python packages:
  ```
  pandas>=1.5.0
  numpy>=1.21.0
  torch>=2.0.0
  matplotlib>=3.5.0
  yfinance>=0.2.0
  beautifulsoup4>=4.9.0
  requests>=2.25.0
  scikit-learn>=1.0.0
  wrds>=3.1.0
  openai>=0.27.0
  python-dotenv>=0.19.0
  ```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd MoE
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   export WRDS_USERNAME="your-username"
   ```

## Usage

1. Run the main script:
   ```bash
   python main.py
   ```

2. The script will:
   - Load and preprocess market data
   - Initialize expert models
   - Train the MoE model
   - Generate predictions
   - Create performance visualizations
   - Save results to CSV

## Model Details

### Expert Models

- **MacroExpert**
  - Analyzes macroeconomic indicators and FOMC statements
  - Uses historical macro uncertainty data
  - Considers GDP and monetary policy impacts

- **FundamentalExpert**
  - Evaluates company financial metrics
  - Focuses on PE ratios and ROE
  - Provides value-based analysis

- **NewsExpert**
  - Processes company-specific news
  - Uses GPT-4 for sentiment analysis
  - Considers market impact of news events

- **TechnicalExpert**
  - Analyzes price and volume patterns
  - Calculates moving averages
  - Identifies technical indicators

- **RiskExpert**
  - Calculates volatility metrics
  - Assesses market beta
  - Evaluates liquidity conditions

### MoE Architecture

The model uses a neural network-based gating mechanism to dynamically weight expert predictions based on market conditions. The gating network learns to assign higher weights to experts that perform better in specific market contexts.

## Output

The model generates:
- Individual expert predictions
- Combined MoE predictions
- Portfolio performance metrics
- Comparison against baseline strategies
- Detailed prediction explanations

## Notes

- Ensure sufficient historical data is available for accurate predictions
- API rate limits may affect execution time
- Model performance depends on data quality and market conditions
