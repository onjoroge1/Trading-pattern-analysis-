/**
 * Charts.js - Handles chart creation and management for the trading system
 */

// Global variable to store the current timeframe
let currentTimeframe = '5min';

/**
 * Load a stock chart with price and indicator data
 * @param {string} symbol - Stock symbol
 * @param {number} days - Number of days of data to fetch
 * @param {string} timeframe - Optional timeframe (e.g., '5min', '15min', '1hour')
 */
function loadStockChart(symbol, days, timeframe) {
    // Show loading indicator
    const chartCanvas = document.getElementById('priceChart');
    const ctx = chartCanvas.getContext('2d');
    
    // Clear any existing chart
    if (window.priceChart) {
        window.priceChart.destroy();
    }
    
    // Use the provided timeframe or current global timeframe
    const tf = timeframe || currentTimeframe;
    
    // Draw loading text
    ctx.clearRect(0, 0, chartCanvas.width, chartCanvas.height);
    ctx.font = '16px Arial';
    ctx.fillStyle = '#aaa';
    ctx.textAlign = 'center';
    ctx.fillText('Loading chart data...', chartCanvas.width / 2, chartCanvas.height / 2);
    
    // Fetch data from API with timeframe
    fetch(`/api/stock_data/${symbol}?days=${days}&timeframe=${tf}`)
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                showChartError(ctx, chartCanvas, data.error);
                return;
            }
            
            // Also fetch pattern data
            fetch(`/api/pattern_data/${symbol}`)
                .then(response => response.json())
                .then(patternData => {
                    // Create chart with both price and pattern data
                    createPriceChart(chartCanvas, data, patternData);
                })
                .catch(error => {
                    console.error('Error fetching pattern data:', error);
                    // Still create chart even if pattern data fails
                    createPriceChart(chartCanvas, data, []);
                });
        })
        .catch(error => {
            console.error('Error fetching stock data:', error);
            showChartError(ctx, chartCanvas, 'Failed to load chart data. Please try again later.');
        });
}

/**
 * Update chart resolution/timeframe
 * @param {string} timeframe - Timeframe to update to (e.g., '5min', '15min', '1hour')
 */
function updateChartResolution(timeframe) {
    // Update the global timeframe
    currentTimeframe = timeframe;
    
    // Get the current symbol from the page
    const symbol = document.querySelector('h1').textContent.trim().split(' ')[0];
    
    // Highlight active timeframe button
    document.querySelectorAll('.btn-group button[onclick^="updateChartResolution"]').forEach(btn => {
        btn.classList.remove('active', 'btn-secondary');
        btn.classList.add('btn-outline-secondary');
    });
    
    const activeBtn = document.querySelector(`.btn-group button[onclick="updateChartResolution('${timeframe}')"]`);
    if (activeBtn) {
        activeBtn.classList.remove('btn-outline-secondary');
        activeBtn.classList.add('active', 'btn-secondary');
    }
    
    // Reload the chart with current days but new timeframe
    // Use a default of 30 days if we can't determine current days
    const daysMatch = window.location.search.match(/days=(\d+)/);
    const days = daysMatch ? parseInt(daysMatch[1]) : 30;
    
    loadStockChart(symbol, days, timeframe);
}

/**
 * Show an error message on the chart canvas
 * @param {CanvasRenderingContext2D} ctx - Canvas context
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {string} message - Error message to display
 */
function showChartError(ctx, canvas, message) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.font = '16px Arial';
    ctx.fillStyle = '#d9534f';
    ctx.textAlign = 'center';
    ctx.fillText(message, canvas.width / 2, canvas.height / 2);
}

/**
 * Create a candlestick chart with indicators
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} data - Price and indicator data
 * @param {Array} patternData - Pattern detection data
 */
function createPriceChart(canvas, data, patternData) {
    const ctx = canvas.getContext('2d');
    
    // Prepare candlestick data
    const candlestickData = [];
    for (let i = 0; i < data.timestamps.length; i++) {
        candlestickData.push({
            x: data.timestamps[i],
            o: data.prices.open[i],
            h: data.prices.high[i],
            l: data.prices.low[i],
            c: data.prices.close[i]
        });
    }
    
    // Prepare pattern annotations
    const annotations = {};
    if (patternData && patternData.length > 0) {
        patternData.forEach((pattern, index) => {
            // Determine if the pattern is bullish or bearish for positioning
            const isBullish = isBullishPattern(pattern.pattern_type);
            const isBearish = isBearishPattern(pattern.pattern_type);
            
            // Determine symbol and position based on pattern type
            let symbol = '●'; // Default circle
            let position = 'top';
            
            if (isBullish) {
                symbol = '▲'; // Triangle up for bullish
                position = 'bottom'; // Show below candle
            } else if (isBearish) {
                symbol = '▼'; // Triangle down for bearish
                position = 'top'; // Show above candle
            }
            
            annotations[`pattern_${index}`] = {
                type: 'point',
                xValue: pattern.timestamp,
                yValue: 0, // Will be adjusted in beforeDraw
                backgroundColor: getPatternColor(pattern.pattern_type),
                borderColor: 'white',
                borderWidth: 1,
                radius: 7,
                pointStyle: symbol,
                label: {
                    enabled: true,
                    content: getPatternLabel(pattern.pattern_type),
                    position: position,
                    backgroundColor: 'rgba(0,0,0,0.7)',
                    color: 'white',
                    font: {
                        size: 11
                    }
                }
            };
        });
    }
    
    // Prepare datasets
    const datasets = [{
        label: 'Price',
        data: candlestickData,
        type: 'candlestick',
        color: {
            up: 'rgba(75, 192, 192, 1)',
            down: 'rgba(255, 99, 132, 1)',
            unchanged: 'rgba(100, 100, 100, 1)',
        }
    }];
    
    // Add RSI if available
    if (data.indicators && data.indicators.rsi) {
        datasets.push({
            label: 'RSI',
            data: data.indicators.rsi,
            borderColor: 'rgba(153, 102, 255, 1)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            borderWidth: 1,
            type: 'line',
            yAxisID: 'rsi'
        });
    }
    
    // Add moving averages if available
    if (data.indicators) {
        for (const key in data.indicators) {
            if (key.startsWith('ma_')) {
                const period = key.split('_')[1];
                const color = getMAColor(period);
                datasets.push({
                    label: `MA ${period}`,
                    data: data.indicators[key],
                    borderColor: color,
                    borderWidth: 1.5,
                    type: 'line',
                    pointRadius: 0,
                    fill: false
                });
            }
        }
    }
    
    // Create chart
    window.priceChart = new Chart(ctx, {
        type: 'line', // Base type, will be overridden by individual datasets
        data: {
            labels: data.timestamps,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Price'
                    }
                },
                rsi: {
                    type: 'linear',
                    display: data.indicators && data.indicators.rsi ? true : false,
                    position: 'right',
                    min: 0,
                    max: 100,
                    grid: {
                        drawOnChartArea: false
                    },
                    title: {
                        display: true,
                        text: 'RSI'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                },
                annotation: {
                    annotations: annotations
                }
            }
        },
        plugins: [{
            id: 'patternLocator',
            beforeDraw: function(chart) {
                // Adjust y values for pattern annotations based on price
                if (patternData && patternData.length > 0) {
                    patternData.forEach((pattern, index) => {
                        const annotation = chart.options.plugins.annotation.annotations[`pattern_${index}`];
                        if (annotation) {
                            // Find the closest data point to the pattern timestamp
                            const timestampIndex = data.timestamps.findIndex(ts => ts === pattern.timestamp);
                            
                            if (timestampIndex !== -1) {
                                // Position based on whether pattern is bullish or bearish
                                if (isBullishPattern(pattern.pattern_type)) {
                                    // For bullish patterns, position below the low price
                                    annotation.yValue = data.prices.low[timestampIndex] * 0.99;
                                } else if (isBearishPattern(pattern.pattern_type)) {
                                    // For bearish patterns, position above the high price
                                    annotation.yValue = data.prices.high[timestampIndex] * 1.01;
                                } else {
                                    // For neutral patterns like doji, position at the closing price
                                    annotation.yValue = data.prices.close[timestampIndex];
                                }
                            }
                        }
                    });
                }
            }
        }]
    });
}

/**
 * Get a color for a specific pattern type
 * @param {string} patternType - Type of pattern
 * @returns {string} Color code
 */
function getPatternColor(patternType) {
    switch (patternType) {
        // Basic patterns
        case 'doji':
            return '#17a2b8'; // Info color
        case 'hammer':
            return '#ffc107'; // Warning color
        case 'consecutive_bullish':
            return '#28a745'; // Success color
        case 'consecutive_bearish':
            return '#dc3545'; // Danger color
            
        // Engulfing patterns
        case 'bullish_engulfing':
            return '#28a745'; // Success color (green)
        case 'bearish_engulfing':
            return '#dc3545'; // Danger color (red)
            
        // Star patterns
        case 'morning_star':
            return '#28a745'; // Success color (green)
        case 'evening_star':
            return '#dc3545'; // Danger color (red)
            
        // Shooting star pattern (bearish)
        case 'shooting_star':
            return '#dc3545'; // Danger color (red)
            
        // Piercing patterns
        case 'piercing_line':
            return '#28a745'; // Success color (green)
        case 'dark_cloud_cover':
            return '#dc3545'; // Danger color (red)
            
        // Three candle patterns
        case 'three_white_soldiers':
            return '#28a745'; // Success color (green)
        case 'three_black_crows':
            return '#dc3545'; // Danger color (red)
            
        default:
            return '#6c757d'; // Secondary color (gray)
    }
}

/**
 * Get a label for a specific pattern type
 * @param {string} patternType - Type of pattern
 * @returns {string} Label text
 */
function getPatternLabel(patternType) {
    switch (patternType) {
        // Basic patterns
        case 'doji':
            return 'Doji';
        case 'hammer':
            return 'Hammer';
        case 'consecutive_bullish':
            return 'Bullish';
        case 'consecutive_bearish':
            return 'Bearish';
            
        // Engulfing patterns
        case 'bullish_engulfing':
            return 'Bull Engulf';
        case 'bearish_engulfing':
            return 'Bear Engulf';
            
        // Star patterns
        case 'morning_star':
            return 'Morning ★';
        case 'evening_star':
            return 'Evening ★';
            
        // Shooting star pattern
        case 'shooting_star':
            return 'Shooting ★';
            
        // Piercing patterns
        case 'piercing_line':
            return 'Piercing';
        case 'dark_cloud_cover':
            return 'Dark Cloud';
            
        // Three candle patterns
        case 'three_white_soldiers':
            return '3 Soldiers';
        case 'three_black_crows':
            return '3 Crows';
            
        default:
            // Convert snake_case to Title Case
            return patternType.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
    }
}

/**
 * Check if a pattern type is bullish
 * @param {string} patternType - The type of pattern 
 * @returns {boolean} True if the pattern is bullish
 */
function isBullishPattern(patternType) {
    const bullishPatterns = [
        'consecutive_bullish', 
        'bullish_engulfing', 
        'morning_star', 
        'piercing_line', 
        'three_white_soldiers',
        'hammer'  // Hammer is typically bullish
    ];
    
    return bullishPatterns.includes(patternType);
}

/**
 * Check if a pattern type is bearish
 * @param {string} patternType - The type of pattern
 * @returns {boolean} True if the pattern is bearish
 */
function isBearishPattern(patternType) {
    const bearishPatterns = [
        'consecutive_bearish', 
        'bearish_engulfing', 
        'evening_star', 
        'shooting_star', 
        'dark_cloud_cover', 
        'three_black_crows'
    ];
    
    return bearishPatterns.includes(patternType);
}

/**
 * Get a color for a moving average based on period
 * @param {number} period - MA period
 * @returns {string} Color code
 */
function getMAColor(period) {
    const periods = {
        20: 'rgba(255, 159, 64, 1)', // Orange
        50: 'rgba(54, 162, 235, 1)', // Blue
        200: 'rgba(153, 102, 255, 1)' // Purple
    };
    
    return periods[period] || 'rgba(201, 203, 207, 1)'; // Gray default
}
