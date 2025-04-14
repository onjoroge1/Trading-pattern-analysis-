/**
 * Trading.js - Handles trading and market data functionality
 */

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
});

/**
 * Refresh market status data
 */
function refreshMarketStatus() {
    const marketStatusContainer = document.querySelector('.card:has(.fas.fa-clock)');
    if (!marketStatusContainer) return;
    
    const marketStatusBody = marketStatusContainer.querySelector('.card-body');
    
    fetch('/api/account_info')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error refreshing market status:', data.error);
                return;
            }
            
            if (window.market_status_refreshed) {
                // Add a subtle indicator that data has been refreshed
                const refreshIndicator = document.createElement('div');
                refreshIndicator.className = 'text-muted text-end small';
                refreshIndicator.innerHTML = '<i class="fas fa-sync-alt me-1"></i>Updated just now';
                
                if (marketStatusBody.querySelector('.text-muted.text-end')) {
                    marketStatusBody.querySelector('.text-muted.text-end').remove();
                }
                
                marketStatusBody.appendChild(refreshIndicator);
            } else {
                window.market_status_refreshed = true;
            }
        })
        .catch(error => {
            console.error('Error fetching market status:', error);
        });
}

/**
 * Refresh account information
 */
function refreshAccountInfo() {
    const accountInfoContainer = document.querySelector('.card:has(.fas.fa-wallet)');
    if (!accountInfoContainer) return;
    
    const accountInfoBody = accountInfoContainer.querySelector('.card-body');
    
    fetch('/api/account_info')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error refreshing account info:', data.error);
                return;
            }
            
            // Update the values
            const equityEl = accountInfoBody.querySelector('.text-primary:nth-of-type(1)');
            const buyingPowerEl = accountInfoBody.querySelector('.text-primary:nth-of-type(2)');
            const cashEl = accountInfoBody.querySelector('.text-primary:nth-of-type(3)');
            const statusEl = accountInfoBody.querySelector('.text-primary:nth-of-type(4)');
            
            if (equityEl) equityEl.textContent = `$${parseFloat(data.equity).toFixed(2)}`;
            if (buyingPowerEl) buyingPowerEl.textContent = `$${parseFloat(data.buying_power).toFixed(2)}`;
            if (cashEl) cashEl.textContent = `$${parseFloat(data.cash).toFixed(2)}`;
            if (statusEl) statusEl.textContent = data.status;
            
            if (window.account_info_refreshed) {
                // Add a subtle indicator that data has been refreshed
                const refreshIndicator = document.createElement('div');
                refreshIndicator.className = 'text-muted text-end small';
                refreshIndicator.innerHTML = '<i class="fas fa-sync-alt me-1"></i>Updated just now';
                
                if (accountInfoBody.querySelector('.text-muted.text-end')) {
                    accountInfoBody.querySelector('.text-muted.text-end').remove();
                }
                
                accountInfoBody.appendChild(refreshIndicator);
            } else {
                window.account_info_refreshed = true;
            }
        })
        .catch(error => {
            console.error('Error fetching account info:', error);
        });
}

/**
 * Refresh positions data
 */
function refreshPositions() {
    const positionsContainer = document.getElementById('refresh-positions').closest('.card');
    if (!positionsContainer) return;
    
    const positionsBody = positionsContainer.querySelector('.card-body');
    const refreshButton = document.getElementById('refresh-positions');
    
    // Show loading state
    if (refreshButton) {
        refreshButton.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Refreshing...';
        refreshButton.disabled = true;
    }
    
    fetch('/api/positions')
        .then(response => response.json())
        .then(data => {
            // Reset button state
            if (refreshButton) {
                refreshButton.innerHTML = '<i class="fas fa-sync-alt me-2"></i>Refresh Positions';
                refreshButton.disabled = false;
            }
            
            if (data.error) {
                console.error('Error refreshing positions:', data.error);
                
                // Show error message
                positionsBody.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error refreshing positions: ${data.error}
                    </div>
                    <div class="d-flex justify-content-end mt-3">
                        <button class="btn btn-outline-primary" id="refresh-positions">
                            <i class="fas fa-sync-alt me-2"></i>
                            Refresh Positions
                        </button>
                    </div>
                `;
                
                // Re-attach event listener
                document.getElementById('refresh-positions').addEventListener('click', refreshPositions);
                return;
            }
            
            if (data.length === 0) {
                positionsBody.innerHTML = `
                    <div class="alert alert-info">
                        No open positions.
                    </div>
                    <div class="d-flex justify-content-end mt-3">
                        <button class="btn btn-outline-primary" id="refresh-positions">
                            <i class="fas fa-sync-alt me-2"></i>
                            Refresh Positions
                        </button>
                    </div>
                `;
                
                // Re-attach event listener
                document.getElementById('refresh-positions').addEventListener('click', refreshPositions);
                return;
            }
            
            // Build positions table
            let tableHtml = `
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Quantity</th>
                                <th>Side</th>
                                <th>Avg Entry</th>
                                <th>Current Price</th>
                                <th>Market Value</th>
                                <th>Unrealized P&L</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            data.forEach(position => {
                tableHtml += `
                    <tr>
                        <td>
                            <a href="/stock/${position.symbol}">
                                ${position.symbol}
                            </a>
                        </td>
                        <td>${position.qty}</td>
                        <td>
                            ${position.side === 'long' 
                                ? '<span class="badge bg-success">Long</span>' 
                                : '<span class="badge bg-danger">Short</span>'}
                        </td>
                        <td>$${parseFloat(position.avg_entry_price).toFixed(2)}</td>
                        <td>$${parseFloat(position.current_price).toFixed(2)}</td>
                        <td>$${parseFloat(position.market_value).toFixed(2)}</td>
                        <td>
                            ${position.unrealized_pl > 0 
                                ? `<span class="text-success">+$${parseFloat(position.unrealized_pl).toFixed(2)}</span>` 
                                : `<span class="text-danger">-$${parseFloat(-position.unrealized_pl).toFixed(2)}</span>`}
                        </td>
                    </tr>
                `;
            });
            
            tableHtml += `
                        </tbody>
                    </table>
                </div>
                <div class="d-flex justify-content-between align-items-center mt-3">
                    <div class="text-muted small">
                        <i class="fas fa-sync-alt me-1"></i>Updated just now
                    </div>
                    <button class="btn btn-outline-primary" id="refresh-positions">
                        <i class="fas fa-sync-alt me-2"></i>
                        Refresh Positions
                    </button>
                </div>
            `;
            
            // Update the content
            positionsBody.innerHTML = tableHtml;
            
            // Re-attach event listener
            document.getElementById('refresh-positions').addEventListener('click', refreshPositions);
        })
        .catch(error => {
            console.error('Error fetching positions:', error);
            
            // Reset button state
            if (refreshButton) {
                refreshButton.innerHTML = '<i class="fas fa-sync-alt me-2"></i>Refresh Positions';
                refreshButton.disabled = false;
            }
            
            // Show error message
            const errorDiv = document.createElement('div');
            errorDiv.className = 'alert alert-danger mt-3';
            errorDiv.innerHTML = `
                <i class="fas fa-exclamation-circle me-2"></i>
                Error refreshing positions: ${error.message}
            `;
            
            if (!positionsBody.querySelector('.alert-danger')) {
                positionsBody.insertBefore(errorDiv, positionsBody.firstChild);
                
                // Auto-remove after 5 seconds
                setTimeout(() => {
                    if (errorDiv.parentNode === positionsBody) {
                        positionsBody.removeChild(errorDiv);
                    }
                }, 5000);
            }
        });
}

/**
 * Load recent trading signals
 */
function loadRecentSignals() {
    const signalsContainer = document.getElementById('recent-signals-container');
    if (!signalsContainer) return;
    
    // Get a random selection of stocks to show signals for
    // In a real app, this would fetch from an endpoint
    const demoStocks = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA'];
    const fetchPromises = demoStocks.map(symbol => 
        fetch(`/api/signal_data/${symbol}`)
            .then(response => response.json())
            .catch(error => {
                console.error(`Error fetching signals for ${symbol}:`, error);
                return [];
            })
    );
    
    Promise.all(fetchPromises)
        .then(results => {
            // Flatten and sort by timestamp (newest first)
            const allSignals = results
                .flat()
                .filter(signal => signal && !signal.error)
                .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
                .slice(0, 10); // Get most recent 10
            
            if (allSignals.length === 0) {
                signalsContainer.innerHTML = `
                    <div class="alert alert-info">
                        <i class="fas fa-info-circle me-2"></i>
                        No recent signals available. Generate signals to see them here.
                    </div>
                `;
                return;
            }
            
            // Build signals table
            let tableHtml = `
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Timestamp</th>
                                <th>Signal</th>
                                <th>Price</th>
                                <th>Confidence</th>
                                <th>Action</th>
                            </tr>
                        </thead>
                        <tbody>
            `;
            
            allSignals.forEach(signal => {
                let signalBadge = '';
                
                switch(signal.signal_type) {
                    case 'buy':
                        signalBadge = '<span class="badge bg-success">Buy</span>';
                        break;
                    case 'short':
                        signalBadge = '<span class="badge bg-danger">Short</span>';
                        break;
                    case 'sell':
                        signalBadge = '<span class="badge bg-danger">Sell</span>';
                        break;
                    default:
                        signalBadge = `<span class="badge bg-secondary">${signal.signal_type}</span>`;
                }
                
                let confidenceBgClass = '';
                if (signal.signal_type === 'buy') {
                    confidenceBgClass = 'bg-success';
                } else if (signal.signal_type === 'short' || signal.signal_type === 'sell') {
                    confidenceBgClass = 'bg-danger';
                } else {
                    confidenceBgClass = 'bg-secondary';
                }
                
                tableHtml += `
                    <tr>
                        <td>${signal.symbol || demoStocks[Math.floor(Math.random() * demoStocks.length)]}</td>
                        <td>${signal.timestamp}</td>
                        <td>${signalBadge}</td>
                        <td>$${parseFloat(signal.price_at_signal).toFixed(2)}</td>
                        <td>
                            <div class="progress" style="height: 20px;">
                                <div class="progress-bar ${confidenceBgClass}" 
                                     role="progressbar" 
                                     style="width: ${(signal.confidence * 100).toFixed(0)}%;" 
                                     aria-valuenow="${(signal.confidence * 100).toFixed(0)}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                    ${(signal.confidence * 100).toFixed(0)}%
                                </div>
                            </div>
                        </td>
                        <td>
                            <form action="/execute_trade" method="post">
                                <input type="hidden" name="signal_id" value="${signal.id}">
                                <button type="submit" class="btn btn-sm btn-primary">
                                    <i class="fas fa-check-circle"></i> Execute
                                </button>
                            </form>
                        </td>
                    </tr>
                `;
            });
            
            tableHtml += `
                        </tbody>
                    </table>
                </div>
            `;
            
            // Update the content
            signalsContainer.innerHTML = tableHtml;
        })
        .catch(error => {
            console.error('Error loading recent signals:', error);
            signalsContainer.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    Error loading recent signals: ${error.message}
                </div>
            `;
        });
}
