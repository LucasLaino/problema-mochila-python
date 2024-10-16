import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

tickers = ['AAPL', 'GOOGL', 'MSFT']
data = yf.download(tickers, period='1y')['Adj Close']

# Calculate asset returns and risk
asset_returns = data.pct_change().mean()  # Renamed to avoid conflict
risk = data.pct_change().std()

covariance_matrix = data.pct_change().cov()

def portfolio_return(weights, returns):
    return np.dot(weights, returns)

def portfolio_risk(weights, covariance_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

def objective_function(weights, returns, covariance_matrix, risk_tolerance):
    risk = portfolio_risk(weights, covariance_matrix)
    expected_return = portfolio_return(weights, returns)
    return -expected_return + 10 * max(0, risk - risk_tolerance)

n_assets = len(tickers)
initial_weights = np.ones(n_assets) / n_assets
bounds = [(0, 1) for _ in range(n_assets)]
constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}

risk_tolerance = 0.05  # Example value, adjust as needed

result = minimize(objective_function, initial_weights, 
                  args=(asset_returns, covariance_matrix, risk_tolerance),  # Pass asset_returns
                  method='SLSQP', bounds=bounds, constraints=constraints)

num_portfolios = 1000
risks = []
portfolio_returns = []  # Renamed to avoid conflict

for _ in range(num_portfolios):
    weights = np.random.random(n_assets)
    weights /= np.sum(weights) 
    risks.append(portfolio_risk(weights, covariance_matrix))
    portfolio_returns.append(portfolio_return(weights, asset_returns)) # Use asset_returns

risks = np.array(risks)
portfolio_returns = np.array(portfolio_returns) # Use portfolio_returns


plt.plot(risks, portfolio_returns, 'o') # Use portfolio_returns
plt.xlabel('Risco')
plt.ylabel('Retorno')
plt.title('Fronteira Eficiente')
plt.show()


plt.pie(result.x, labels=tickers, autopct='%1.1f%%')
plt.title('Alocação de Ativos')
plt.show()