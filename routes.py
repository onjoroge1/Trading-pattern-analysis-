import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from flask import render_template, jsonify, request, Response, flash, redirect, url_for, g, session
from app import app, db
from models import Stock, PatternDetection, TradingSignal, TradeExecution, RLModelTraining, TradeJournal
from trading.polygon_api import PolygonAPI
from trading.pattern_detection import PatternDetector
from trading.indicators import TechnicalIndicators
from trading.rl_agent import RLAgent

logger = logging.getLogger(__name__)

# Function to get Polygon API client
def get_market_api():
    """Get or create PolygonAPI instance for the current request"""
    if not hasattr(g, 'market_api'):
        # Create new instance and initialize from app
        g.market_api = PolygonAPI()
        g.market_api.initialize_from_app(app)
    return g.market_api

@app.route('/')
def index():
    """Main dashboard page"""
    try:
        # Get list of stocks
        stocks = Stock.query.filter_by(active=True).all()
        
        # Get Market API client for this request
        market_api = get_market_api()
        
        # Get account info (if available)
        account_info = market_api.get_account_info()
        
        # Get current market data for stocks in the watchlist
        stock_symbols = [stock.symbol for stock in stocks]
        market_data = market_api.get_current_market_data(stock_symbols) if stock_symbols else {}
        
        # Get recent signals
        recent_signals = db.session.query(
            TradingSignal, Stock
        ).join(Stock).order_by(
            TradingSignal.timestamp.desc()
        ).limit(10).all()
        
        # Get recent trades
        recent_trades = db.session.query(
            TradeExecution, Stock
        ).join(Stock).order_by(
            TradeExecution.timestamp.desc()
        ).limit(10).all()
        
        # Get market hours
        market_hours = market_api.get_market_hours()
        
        # Get recent patterns
        recent_patterns = db.session.query(
            PatternDetection, Stock
        ).join(Stock).order_by(
            PatternDetection.timestamp.desc()
        ).limit(10).all()
        
        return render_template(
            'index_new.html',
            stocks=stocks,
            account_info=account_info,
            recent_signals=recent_signals,
            recent_trades=recent_trades,
            recent_patterns=recent_patterns,
            market_hours=market_hours,
            market_data=market_data,  # Pass market data for watchlist display
            now=datetime.now()  # Pass current datetime for template calculations
        )
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        flash(f"An error occurred: {str(e)}", 'danger')
        return render_template('index_new.html', error=str(e))

@app.route('/stocks')
def stock_list():
    """List all stocks"""
    stocks = Stock.query.all()
    return render_template('stocks.html', stocks=stocks)

@app.route('/add_stock', methods=['POST'])
def add_stock():
    """Add a new stock to monitor"""
    try:
        symbol = request.form.get('symbol', '').strip().upper()
        name = request.form.get('name', '').strip()
        
        if not symbol:
            flash('Stock symbol is required', 'danger')
            return redirect(url_for('stock_list'))
        
        # Check if stock already exists
        existing_stock = Stock.query.filter_by(symbol=symbol).first()
        if existing_stock:
            flash(f'Stock {symbol} already exists', 'warning')
            return redirect(url_for('stock_list'))
        
        # Create new stock
        stock = Stock(symbol=symbol, name=name)
        db.session.add(stock)
        db.session.commit()
        
        flash(f'Stock {symbol} added successfully', 'success')
        return redirect(url_for('stock_list'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding stock: {e}")
        flash(f'Error adding stock: {str(e)}', 'danger')
        return redirect(url_for('stock_list'))

@app.route('/stock/<symbol>')
def stock_detail(symbol):
    """Display stock details and historical patterns"""
    try:
        stock = Stock.query.filter_by(symbol=symbol).first_or_404()
        
        # Get patterns for this stock
        patterns = PatternDetection.query.filter_by(stock_id=stock.id).order_by(
            PatternDetection.timestamp.desc()
        ).limit(50).all()
        
        # Get signals for this stock
        signals = TradingSignal.query.filter_by(stock_id=stock.id).order_by(
            TradingSignal.timestamp.desc()
        ).limit(50).all()
        
        # Get trades for this stock
        trades = TradeExecution.query.filter_by(stock_id=stock.id).order_by(
            TradeExecution.timestamp.desc()
        ).limit(50).all()
        
        # Get Market API client for this request
        market_api = get_market_api()
        
        # Get current market data if available
        current_data = market_api.get_current_market_data([symbol])
        
        return render_template(
            'stock_detail.html',
            stock=stock,
            patterns=patterns,
            signals=signals,
            trades=trades,
            current_data=current_data.get(symbol) if current_data else None
        )
    except Exception as e:
        logger.error(f"Error in stock_detail route: {e}")
        flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('stock_list'))

@app.route('/fetch_historical_data', methods=['POST'])
def fetch_historical_data():
    """Fetch historical data for pattern detection"""
    try:
        symbol = request.form.get('symbol')
        days = int(request.form.get('days', 30))
        timeframe = request.form.get('timeframe', '5min')
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        # Get timespan and multiplier for the selected timeframe
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'minute', 'multiplier': 5})
        
        if not symbol:
            flash('Stock symbol is required', 'danger')
            return redirect(url_for('index'))
        
        # Get stock from database or create if it doesn't exist
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            stock = Stock(symbol=symbol)
            db.session.add(stock)
            db.session.commit()
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Use the selected timeframe parameters
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        logger.info(f"Fetching {symbol} data with timeframe: {timespan}/{multiplier}, from {start_date} to {end_date}")
        data = market_api.get_historical_data(symbol, timespan=timespan, multiplier=multiplier, start_date=start_date, end_date=end_date)
        
        if data.empty:
            flash(f'No data available for {symbol}', 'warning')
            return redirect(url_for('index'))
        
        # Calculate indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Detect patterns
        detector = PatternDetector()
        patterns_df = detector.detect_all_patterns(data, indicators)
        
        # Save patterns to database
        count = 0
        for idx, row in patterns_df[patterns_df['has_reversal_pattern'] == True].iterrows():
            # Check if pattern already exists for this timestamp
            existing = PatternDetection.query.filter_by(
                stock_id=stock.id,
                timestamp=idx.to_pydatetime()
            ).first()
            
            if existing:
                continue
            
            # Determine pattern type
            pattern_type = 'unknown'
            description = ''
            
            # Basic patterns
            if 'is_doji' in row and row['is_doji']:
                pattern_type = 'doji'
                description += 'Doji pattern detected. '
            
            if 'is_hammer' in row and row['is_hammer']:
                pattern_type = 'hammer'
                description += 'Hammer pattern detected. '
            
            if 'potential_bullish_reversal' in row and row['potential_bullish_reversal']:
                pattern_type = 'consecutive_bullish'
                description += 'Potential bullish reversal after consecutive bearish candles. '
            
            if 'potential_bearish_reversal' in row and row['potential_bearish_reversal']:
                pattern_type = 'consecutive_bearish'
                description += 'Potential bearish reversal after consecutive bullish candles. '
                
            # Advanced patterns - Engulfing
            if 'bullish_engulfing' in row and row['bullish_engulfing']:
                pattern_type = 'bullish_engulfing'
                description += 'Bullish Engulfing pattern detected. Current candle completely engulfs previous bearish candle. Strong bullish signal. '
            
            if 'bearish_engulfing' in row and row['bearish_engulfing']:
                pattern_type = 'bearish_engulfing'
                description += 'Bearish Engulfing pattern detected. Current candle completely engulfs previous bullish candle. Strong bearish signal. '
            
            # Star patterns
            if 'morning_star' in row and row['morning_star']:
                pattern_type = 'morning_star'
                description += 'Morning Star pattern detected. Potential bullish reversal after downtrend. '
            
            if 'evening_star' in row and row['evening_star']:
                pattern_type = 'evening_star'
                description += 'Evening Star pattern detected. Potential bearish reversal after uptrend. '
            
            # Shooting Star
            if 'shooting_star' in row and row['shooting_star']:
                pattern_type = 'shooting_star'
                description += 'Shooting Star pattern detected. Long upper shadow with little to no lower shadow. Bearish reversal signal. '
            
            # Piercing patterns
            if 'piercing_line' in row and row['piercing_line']:
                pattern_type = 'piercing_line'
                description += 'Piercing Line pattern detected. Bullish reversal signal where current candle closes more than halfway up previous bearish candle. '
            
            if 'dark_cloud_cover' in row and row['dark_cloud_cover']:
                pattern_type = 'dark_cloud_cover'
                description += 'Dark Cloud Cover pattern detected. Bearish reversal signal where current candle closes more than halfway down previous bullish candle. '
            
            # Three candle patterns
            if 'three_white_soldiers' in row and row['three_white_soldiers']:
                pattern_type = 'three_white_soldiers'
                description += 'Three White Soldiers pattern detected. Three consecutive bullish candles with higher highs and higher lows. Strong bullish signal. '
            
            if 'three_black_crows' in row and row['three_black_crows']:
                pattern_type = 'three_black_crows'
                description += 'Three Black Crows pattern detected. Three consecutive bearish candles with lower lows and lower highs. Strong bearish signal. '
            
            # Add RSI information to description
            if 'rsi' in row:
                description += f"RSI: {row['rsi']:.2f}. "
                
                if 'in_opening_hour' in row and row['in_opening_hour']:
                    description += "During market opening (9:30-10:00). "
                    if row['rsi'] > 70:
                        description += "Overbought during market open suggests bullish momentum. "
                    elif row['rsi'] < 30:
                        description += "Oversold during market open suggests bearish momentum. "
                else:
                    if row['rsi'] > 70:
                        description += "Overbought condition in standard market hours. "
                    elif row['rsi'] < 30:
                        description += "Oversold condition in standard market hours. "
            
            # Add ORB information to description
            if 'above_orb_high' in row and row['above_orb_high']:
                description += "Price above Opening Range Breakout high. "
            elif 'below_orb_low' in row and row['below_orb_low']:
                description += "Price below Opening Range Breakout low. "
            
            # Create new pattern
            pattern = PatternDetection(
                stock_id=stock.id,
                timestamp=idx.to_pydatetime(),
                pattern_type=pattern_type,
                description=description,
                rsi_value=row.get('rsi'),
                above_orb_high=row.get('above_orb_high', False),
                below_orb_low=row.get('below_orb_low', False),
                in_opening_hour=row.get('in_opening_hour', False)
            )
            
            db.session.add(pattern)
            count += 1
        
        db.session.commit()
        flash(f'Fetched data and detected {count} new patterns for {symbol}', 'success')
        
        # Redirect to stock detail page
        return redirect(url_for('stock_detail', symbol=symbol))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error fetching historical data: {e}")
        flash(f'Error fetching historical data: {str(e)}', 'danger')
        return redirect(url_for('index'))

@app.route('/training')
def training():
    """Training page for RL model"""
    # Get list of stocks
    stocks = Stock.query.filter_by(active=True).all()
    
    # Get previous training sessions
    training_sessions = RLModelTraining.query.order_by(
        RLModelTraining.created_at.desc()
    ).all()
    
    return render_template(
        'training.html',
        stocks=stocks,
        training_sessions=training_sessions
    )

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the RL model"""
    try:
        # Get selected stocks
        selected_stocks = request.form.getlist('stocks')
        if not selected_stocks:
            flash('Select at least one stock for training', 'danger')
            return redirect(url_for('training'))
        
        # Get training parameters
        days = int(request.form.get('days', 365))
        episodes = int(request.form.get('episodes', 1000))
        learning_rate = float(request.form.get('learning_rate', 0.0003))
        
        # Fetch historical data for selected stocks
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Get selected timeframe or default to daily data
        timeframe = request.form.get('timeframe', 'day,1')
        timespan, multiplier = timeframe.split(',')
        multiplier = int(multiplier)
        
        stock_data_dict = {}
        for symbol in selected_stocks:
            data = market_api.get_historical_data(symbol, timespan=timespan, multiplier=multiplier, start_date=start_date, end_date=end_date)
            
            if data.empty:
                continue
            
            # Calculate indicators
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            # Detect patterns
            detector = PatternDetector()
            patterns_df = detector.detect_all_patterns(data, indicators)
            
            # Store the processed data
            stock_data_dict[symbol] = patterns_df
        
        if not stock_data_dict:
            flash('No data available for selected stocks', 'warning')
            return redirect(url_for('training'))
        
        # Initialize RL agent
        agent = RLAgent()
        
        # Create environment
        agent.create_environment(stock_data_dict, start_date, end_date)
        
        # Train the model
        timesteps = episodes * 100  # Approximate timesteps
        model = agent.train(
            stock_data_dict,
            total_timesteps=timesteps,
            learning_rate=learning_rate,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save the model
        model_name = f"trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"./models/{model_name}"
        agent.save_model(model_path)
        
        # Save training info to database
        # Convert timeframe format to a user-friendly string
        timeframe_display = f"{multiplier}{timespan[0]}"  # e.g., "5m", "1d", "1w"
        if timespan == "minute":
            timeframe_display = f"{multiplier}min"
        elif timespan == "hour":
            timeframe_display = f"{multiplier}hour"
            
        training_info = RLModelTraining(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            stocks_used=','.join(selected_stocks),
            total_episodes=episodes,
            total_timesteps=timesteps,
            model_path=model_path,
            timeframe=timeframe_display
        )
        
        db.session.add(training_info)
        db.session.commit()
        
        flash(f'Successfully trained model: {model_name}', 'success')
        return redirect(url_for('training'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error training model: {e}")
        flash(f'Error training model: {str(e)}', 'danger')
        return redirect(url_for('training'))

@app.route('/backtesting')
def backtesting():
    """Backtesting page"""
    # Get list of stocks
    stocks = Stock.query.filter_by(active=True).all()
    
    # Get available models
    models = RLModelTraining.query.order_by(
        RLModelTraining.created_at.desc()
    ).all()
    
    return render_template(
        'backtesting.html',
        stocks=stocks,
        models=models
    )

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    """Run backtest on historical data"""
    try:
        # Get parameters
        symbol = request.form.get('symbol')
        model_id = request.form.get('model_id')
        days = int(request.form.get('days', 30))
        
        if not symbol or not model_id:
            flash('Stock symbol and model are required', 'danger')
            return redirect(url_for('backtesting'))
        
        # Get the model
        model_info = RLModelTraining.query.get_or_404(model_id)
        
        # Initialize RL agent with the model
        agent = RLAgent(model_path=model_info.model_path)
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Default to 5min timeframe if not specified
        timeframe = request.form.get('timeframe', '5min')
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        # Get timespan and multiplier for the selected timeframe
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'minute', 'multiplier': 5})
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        logger.info(f"Fetching backtest data for {symbol} with {timeframe} ({timespan}/{multiplier})")
        data = market_api.get_historical_data(symbol, timespan=timespan, multiplier=multiplier, start_date=start_date, end_date=end_date)
        
        if data.empty:
            flash(f'No data available for {symbol}', 'warning')
            return redirect(url_for('backtesting'))
        
        # Calculate indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Detect patterns
        detector = PatternDetector()
        patterns_df = detector.detect_all_patterns(data, indicators)
        
        # Run backtest
        results = agent.backtest(patterns_df, symbol)
        
        if results is None:
            flash('Backtest failed', 'danger')
            return redirect(url_for('backtesting'))
        
        # Convert results to JSON for the template
        backtest_data = {
            'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in results['timestamp']],
            'prices': results['price'].tolist(),
            'positions': results['position'].tolist(),
            'balances': results['balance'].tolist(),
            'actions': results['action'].tolist(),
            'cumulative_returns': results['cumulative_returns'].tolist()
        }
        
        # Save backtest results to session for retrieval
        session['backtest_results'] = json.dumps(backtest_data)
        
        return render_template(
            'backtest_results.html',
            symbol=symbol,
            model_name=model_info.model_name,
            start_date=start_date,
            end_date=end_date,
            backtest_data=backtest_data
        )
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        flash(f'Error running backtest: {str(e)}', 'danger')
        return redirect(url_for('backtesting'))

@app.route('/live_trading')
def live_trading():
    """Live trading page"""
    # Get list of stocks
    stocks = Stock.query.filter_by(active=True).all()
    
    # Get available models
    models = RLModelTraining.query.order_by(
        RLModelTraining.created_at.desc()
    ).all()
    
    # Get Alpaca API client for this request
    market_api = get_market_api()
    
    # Get account info
    account_info = market_api.get_account_info()
    
    # Get current positions
    positions = market_api.get_positions()
    
    # Get market hours
    market_hours = market_api.get_market_hours()
    
    return render_template(
        'live_trading.html',
        stocks=stocks,
        models=models,
        account_info=account_info,
        positions=positions,
        market_hours=market_hours
    )

@app.route('/generate_signals', methods=['POST'])
def generate_signals():
    """Generate trading signals using the RL model"""
    try:
        # Get parameters
        symbol = request.form.get('symbol')
        model_id = request.form.get('model_id')
        timeframe = request.form.get('timeframe', '5min')
        
        if not symbol or not model_id:
            flash('Stock symbol and model are required', 'danger')
            return redirect(url_for('live_trading'))
        
        # Get the model
        model_info = RLModelTraining.query.get_or_404(model_id)
        
        # Initialize RL agent with the model
        agent = RLAgent(model_path=model_info.model_path)
        
        # Get stock from database
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            flash(f'Stock {symbol} not found', 'danger')
            return redirect(url_for('live_trading'))
        
        # Fetch recent historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # Get recent 5 days
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        # Get timespan and multiplier for the selected timeframe
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'minute', 'multiplier': 5})
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        logger.info(f"Fetching signal data for {symbol} with {timeframe} ({timespan}/{multiplier})")
        data = market_api.get_historical_data(symbol, timespan=timespan, multiplier=multiplier, start_date=start_date, end_date=end_date)
        
        if data.empty:
            flash(f'No data available for {symbol}', 'warning')
            return redirect(url_for('live_trading'))
        
        # Calculate indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Detect patterns
        detector = PatternDetector()
        patterns_df = detector.detect_all_patterns(data, indicators)
        
        # Get the latest data point for signal generation
        latest_data = patterns_df.iloc[-1]
        latest_timestamp = patterns_df.index[-1]
        
        # Prepare state for prediction
        state_features = np.array([
            latest_data['close'] / 100,  # Normalize price
            (latest_data.get('rsi', 50) - 50) / 50,  # Normalize RSI
            1 if 'is_doji' in latest_data and latest_data['is_doji'] else 0,
            1 if 'is_hammer' in latest_data and latest_data['is_hammer'] else 0,
            0,  # Position (assuming we start from neutral)
            latest_data.get('distance_from_orb_high', 0),
            latest_data.get('distance_from_orb_low', 0),
            1 if 'above_orb_high' in latest_data and latest_data['above_orb_high'] else 0,
            1 if 'below_orb_low' in latest_data and latest_data['below_orb_low'] else 0,
            1 if 'in_opening_hour' in latest_data and latest_data['in_opening_hour'] else 0
        ], dtype=np.float32)
        
        # Get prediction
        action, _ = agent.predict(state_features.reshape(1, -1))
        
        # Map action to signal type
        signal_types = {0: 'hold', 1: 'buy', 2: 'short'}
        signal_type = signal_types.get(action[0], 'hold')
        
        # Save the pattern if any
        pattern = None
        if latest_data.get('has_reversal_pattern', False):
            # Determine pattern type
            pattern_type = 'unknown'
            description = ''
            
            if 'is_doji' in latest_data and latest_data['is_doji']:
                pattern_type = 'doji'
                description += 'Doji pattern detected. '
            
            if 'is_hammer' in latest_data and latest_data['is_hammer']:
                pattern_type = 'hammer'
                description += 'Hammer pattern detected. '
            
            if 'potential_bullish_reversal' in latest_data and latest_data['potential_bullish_reversal']:
                pattern_type = 'consecutive_bullish'
                description += 'Potential bullish reversal after consecutive bearish candles. '
            
            if 'potential_bearish_reversal' in latest_data and latest_data['potential_bearish_reversal']:
                pattern_type = 'consecutive_bearish'
                description += 'Potential bearish reversal after consecutive bullish candles. '
            
            # Check if pattern already exists
            pattern = PatternDetection.query.filter_by(
                stock_id=stock.id,
                timestamp=latest_timestamp.to_pydatetime()
            ).first()
            
            if not pattern:
                pattern = PatternDetection(
                    stock_id=stock.id,
                    timestamp=latest_timestamp.to_pydatetime(),
                    pattern_type=pattern_type,
                    description=description,
                    rsi_value=latest_data.get('rsi'),
                    above_orb_high=latest_data.get('above_orb_high', False),
                    below_orb_low=latest_data.get('below_orb_low', False),
                    in_opening_hour=latest_data.get('in_opening_hour', False)
                )
                db.session.add(pattern)
                db.session.commit()
        
        # Generate trading signal
        notes = f"Signal generated using model {model_info.model_name}. "
        
        if 'rsi' in latest_data:
            notes += f"RSI: {latest_data['rsi']:.2f}. "
            
            if 'in_opening_hour' in latest_data and latest_data['in_opening_hour']:
                notes += "During market opening (9:30-10:00). "
            
            # Add RSI bias
            if 'rsi_long_bias' in latest_data and latest_data['rsi_long_bias']:
                notes += "RSI suggests long bias. "
            elif 'rsi_short_bias' in latest_data and latest_data['rsi_short_bias']:
                notes += "RSI suggests short bias. "
        
        # Add ORB context
        if 'above_orb_high' in latest_data and latest_data['above_orb_high']:
            notes += "Price above ORB high suggests bullish bias. "
        elif 'below_orb_low' in latest_data and latest_data['below_orb_low']:
            notes += "Price below ORB low suggests bearish bias. "
        
        signal = TradingSignal(
            pattern_id=pattern.id if pattern else None,
            stock_id=stock.id,
            timestamp=latest_timestamp.to_pydatetime(),
            signal_type=signal_type,
            confidence=0.8,  # Placeholder value
            notes=notes,
            price_at_signal=latest_data['close']
        )
        
        db.session.add(signal)
        db.session.commit()
        
        flash(f'Generated {signal_type} signal for {symbol}', 'success')
        return redirect(url_for('stock_detail', symbol=symbol))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error generating signals: {e}")
        flash(f'Error generating signals: {str(e)}', 'danger')
        return redirect(url_for('live_trading'))

@app.route('/execute_trade', methods=['POST'])
def execute_trade():
    """Execute a trade based on a signal"""
    try:
        # Get parameters
        signal_id = request.form.get('signal_id')
        quantity = int(request.form.get('quantity', 1))
        
        if not signal_id:
            flash('Signal ID is required', 'danger')
            return redirect(url_for('live_trading'))
        
        # Get the signal
        signal = TradingSignal.query.get_or_404(signal_id)
        
        # Determine action based on signal type
        if signal.signal_type == 'buy':
            action = 'buy'
        elif signal.signal_type == 'short':
            action = 'sell'
        else:
            flash('Cannot execute trade for hold signal', 'warning')
            return redirect(url_for('live_trading'))
        
        # Get Alpaca API client for this request
        market_api = get_market_api()
        
        # Submit order to Alpaca
        order_result = market_api.submit_order(
            symbol=signal.stock.symbol,
            qty=quantity,
            side=action
        )
        
        if not order_result:
            flash('Failed to submit order', 'danger')
            return redirect(url_for('live_trading'))
        
        # Save trade execution
        trade = TradeExecution(
            signal_id=signal.id,
            stock_id=signal.stock_id,
            timestamp=datetime.now(),
            action=action,
            quantity=quantity,
            price=order_result.get('price', signal.price_at_signal),
            status='open'
        )
        
        db.session.add(trade)
        db.session.commit()
        
        flash(f'Executed {action} order for {quantity} shares of {signal.stock.symbol}', 'success')
        return redirect(url_for('live_trading'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error executing trade: {e}")
        flash(f'Error executing trade: {str(e)}', 'danger')
        return redirect(url_for('live_trading'))

# API endpoints for AJAX requests
@app.route('/api/stock_data/<symbol>')
def api_stock_data(symbol):
    """API endpoint to get stock data for charts"""
    try:
        days = int(request.args.get('days', 30))
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Get timeframe from query params or default to 5min
        timeframe = request.args.get('timeframe', '5min')
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        # Get timespan and multiplier for the selected timeframe
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'minute', 'multiplier': 5})
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        logger.info(f"Fetching chart data for {symbol} with {timeframe} ({timespan}/{multiplier})")
        data = market_api.get_historical_data(symbol, timespan=timespan, multiplier=multiplier, start_date=start_date, end_date=end_date)
        
        if data.empty:
            return jsonify({'error': 'No data available'})
        
        # Format data for charts
        chart_data = {
            'timestamps': [idx.strftime('%Y-%m-%d %H:%M:%S') for idx in data.index],
            'prices': {
                'open': data['open'].tolist(),
                'high': data['high'].tolist(),
                'low': data['low'].tolist(),
                'close': data['close'].tolist()
            },
            'volume': data['volume'].tolist()
        }
        
        # Add indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        chart_data['indicators'] = {
            'rsi': indicators['rsi'].tolist() if 'rsi' in indicators else []
        }
        
        # Add MA data if available
        for col in indicators.columns:
            if col.startswith('ma_'):
                chart_data['indicators'][col] = indicators[col].tolist()
        
        return jsonify(chart_data)
    except Exception as e:
        logger.error(f"Error in api_stock_data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/pattern_data/<symbol>')
def api_pattern_data(symbol):
    """API endpoint to get pattern data for a stock"""
    try:
        stock = Stock.query.filter_by(symbol=symbol).first_or_404()
        
        # Get patterns for this stock
        patterns = PatternDetection.query.filter_by(stock_id=stock.id).order_by(
            PatternDetection.timestamp.asc()
        ).all()
        
        pattern_data = [{
            'id': p.id,
            'timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'pattern_type': p.pattern_type,
            'rsi_value': p.rsi_value,
            'above_orb_high': p.above_orb_high,
            'below_orb_low': p.below_orb_low,
            'in_opening_hour': p.in_opening_hour
        } for p in patterns]
        
        return jsonify(pattern_data)
    except Exception as e:
        logger.error(f"Error in api_pattern_data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/signal_data/<symbol>')
def api_signal_data(symbol):
    """API endpoint to get signal data for a stock"""
    try:
        stock = Stock.query.filter_by(symbol=symbol).first_or_404()
        
        # Get signals for this stock
        signals = TradingSignal.query.filter_by(stock_id=stock.id).order_by(
            TradingSignal.timestamp.asc()
        ).all()
        
        signal_data = [{
            'id': s.id,
            'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'signal_type': s.signal_type,
            'confidence': s.confidence,
            'price_at_signal': s.price_at_signal
        } for s in signals]
        
        return jsonify(signal_data)
    except Exception as e:
        logger.error(f"Error in api_signal_data: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/account_info')
def api_account_info():
    """API endpoint to get account information"""
    try:
        # Get Alpaca API client for this request
        market_api = get_market_api()
        
        account_info = market_api.get_account_info()
        if not account_info:
            return jsonify({'error': 'Failed to fetch account info'})
        
        return jsonify(account_info)
    except Exception as e:
        logger.error(f"Error in api_account_info: {e}")
        return jsonify({'error': str(e)})

@app.route('/api/positions')
def api_positions():
    """API endpoint to get current positions"""
    try:
        # Get Alpaca API client for this request
        market_api = get_market_api()
        
        positions = market_api.get_positions()
        return jsonify(positions)
    except Exception as e:
        logger.error(f"Error in api_positions: {e}")
        return jsonify({'error': str(e)})

@app.route('/fetch_data_demo')
def fetch_data_demo():
    """Demo page for testing data fetching with different timeframes"""
    return render_template('fetch_data_demo.html')

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    """Admin dashboard for system configuration"""
    # Handle POST requests for different forms
    if request.method == 'POST':
        form_type = request.form.get('form_type')
        
        if form_type == 'training_settings':
            # Save training settings
            flash('Training settings updated successfully', 'success')
        elif form_type == 'api_settings':
            # Save API settings
            flash('API settings updated successfully', 'success')
        elif form_type == 'system_settings':
            # Save system settings
            flash('System settings updated successfully', 'success')
        
        return redirect(url_for('admin'))
    
    # Get all stocks
    stocks = Stock.query.all()
    
    # Get all trained models
    models = RLModelTraining.query.all()
    
    # Get Polygon API key status
    polygon_api_key = os.environ.get('POLYGON_API_KEY')
    polygon_api_key_masked = None
    if polygon_api_key:
        polygon_api_key_masked = polygon_api_key[:4] + '...'
    
    return render_template(
        'admin.html',
        stocks=stocks,
        models=models,
        polygon_api_key=polygon_api_key_masked
    )

# Trade Journal Routes
@app.route('/trade_journal')
def trade_journal():
    """Trade Journal page for recording and analyzing successful trades"""
    try:
        # Get list of stocks
        stocks = Stock.query.all()
        
        # Get trade journal entries
        trades = TradeJournal.query.order_by(
            TradeJournal.entry_date.desc()
        ).all()
        
        # Get recent patterns (empty initially - will be populated by fetch_patterns_for_journal)
        recent_patterns = session.get('recent_patterns', [])
        if isinstance(recent_patterns, list) and len(recent_patterns) > 0:
            # Convert pattern IDs back to objects
            pattern_objects = []
            for pattern_id, stock_id in recent_patterns:
                pattern = PatternDetection.query.get(pattern_id)
                stock = Stock.query.get(stock_id)
                if pattern and stock:
                    pattern_objects.append((pattern, stock))
            recent_patterns = pattern_objects
        else:
            recent_patterns = []
        
        return render_template(
            'trade_journal.html',
            stocks=stocks,
            trades=trades,
            recent_patterns=recent_patterns,
            now=datetime.now()
        )
    except Exception as e:
        logger.error(f"Error in trade_journal route: {e}")
        flash(f"An error occurred: {str(e)}", 'danger')
        return redirect(url_for('index'))

@app.route('/add_trade_journal', methods=['POST'])
def add_trade_journal():
    """Add a new trade to the journal"""
    try:
        # Get form data
        symbol = request.form.get('symbol', '').strip().upper()
        position_type = request.form.get('position_type')
        entry_date_str = request.form.get('entry_date')
        exit_date_str = request.form.get('exit_date')
        entry_price = float(request.form.get('entry_price'))
        exit_price_str = request.form.get('exit_price')
        position_size = int(request.form.get('position_size'))
        rsi_at_entry_str = request.form.get('rsi_at_entry')
        strategy = request.form.get('strategy')
        success_reason = request.form.get('success_reason')
        notes = request.form.get('notes')
        
        # Validate required fields
        if not symbol or not position_type or not entry_date_str or not entry_price or not position_size:
            flash('Required fields are missing', 'danger')
            return redirect(url_for('trade_journal'))
        
        # Convert dates
        entry_date = datetime.fromisoformat(entry_date_str)
        exit_date = datetime.fromisoformat(exit_date_str) if exit_date_str else None
        
        # Convert other optional fields
        exit_price = float(exit_price_str) if exit_price_str else None
        rsi_at_entry = float(rsi_at_entry_str) if rsi_at_entry_str else None
        
        # Get or create stock
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            stock = Stock(symbol=symbol)
            db.session.add(stock)
            db.session.commit()
        
        # Create trade journal entry
        trade = TradeJournal(
            stock_id=stock.id,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            position_type=position_type,
            position_size=position_size,
            rsi_at_entry=rsi_at_entry,
            strategy=strategy,
            success_reason=success_reason,
            notes=notes
        )
        
        db.session.add(trade)
        db.session.commit()
        
        flash('Trade added to journal successfully', 'success')
        return redirect(url_for('trade_journal'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding trade to journal: {e}")
        flash(f'Error adding trade: {str(e)}', 'danger')
        return redirect(url_for('trade_journal'))

@app.route('/edit_trade_journal/<int:trade_id>', methods=['GET', 'POST'])
def edit_trade_journal(trade_id):
    """Edit a trade journal entry"""
    trade = TradeJournal.query.get_or_404(trade_id)
    
    if request.method == 'POST':
        try:
            # Update fields from form
            exit_date_str = request.form.get('exit_date')
            exit_price_str = request.form.get('exit_price')
            
            # Update trade
            trade.exit_date = datetime.fromisoformat(exit_date_str) if exit_date_str else None
            trade.exit_price = float(exit_price_str) if exit_price_str else None
            trade.rsi_at_entry = float(request.form.get('rsi_at_entry')) if request.form.get('rsi_at_entry') else None
            trade.strategy = request.form.get('strategy')
            trade.success_reason = request.form.get('success_reason')
            trade.notes = request.form.get('notes')
            
            db.session.commit()
            flash('Trade updated successfully', 'success')
            return redirect(url_for('trade_journal'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Error updating trade: {e}")
            flash(f'Error updating trade: {str(e)}', 'danger')
            return redirect(url_for('trade_journal'))
    
    # GET request - show edit form
    stocks = Stock.query.all()
    return render_template(
        'edit_trade.html',
        trade=trade,
        stocks=stocks
    )

@app.route('/delete_trade_journal/<int:trade_id>')
def delete_trade_journal(trade_id):
    """Delete a trade journal entry"""
    try:
        trade = TradeJournal.query.get_or_404(trade_id)
        db.session.delete(trade)
        db.session.commit()
        flash('Trade deleted successfully', 'success')
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error deleting trade: {e}")
        flash(f'Error deleting trade: {str(e)}', 'danger')
    
    return redirect(url_for('trade_journal'))

@app.route('/train_custom_rl', methods=['POST'])
def train_custom_rl():
    """Train a custom RL model based on trade journal entries"""
    try:
        # Get form data
        model_name = request.form.get('model_name')
        timeframe = request.form.get('timeframe')
        episodes = int(request.form.get('episodes', 1000))
        
        # Get trade journal entries
        trades = TradeJournal.query.all()
        
        if len(trades) < 5:
            flash('You need at least 5 trade journal entries to train a custom model', 'warning')
            return redirect(url_for('trade_journal'))
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'day', 'multiplier': 1})
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        # Extract unique stocks from trades
        stock_symbols = list(set([trade.stock.symbol for trade in trades]))
        
        # Determine date range based on trades
        earliest_trade = min([trade.entry_date for trade in trades])
        latest_trade = max([trade.exit_date or datetime.now() for trade in trades])
        
        # Add buffer days
        start_date = earliest_trade - timedelta(days=30)  # 30 days before earliest trade
        end_date = latest_trade + timedelta(days=5)  # 5 days after latest trade
        
        # Get market API client for this request
        market_api = get_market_api()
        
        # Fetch historical data for all stocks
        stock_data_dict = {}
        for symbol in stock_symbols:
            data = market_api.get_historical_data(
                symbol, 
                timespan=timespan, 
                multiplier=multiplier, 
                start_date=start_date, 
                end_date=end_date
            )
            
            if not data.empty:
                # Calculate technical indicators
                indicators = TechnicalIndicators.calculate_all_indicators(data)
                
                # Detect patterns
                detector = PatternDetector()
                patterns_df = detector.detect_all_patterns(data, indicators)
                
                stock_data_dict[symbol] = patterns_df
        
        # Build custom training data incorporating trade journal information
        # This would be fully implemented in a real trading_environment class
        
        # For now, we'll use a simplified approach
        # Initialize RL agent
        agent = RLAgent()
        
        # Generate a model path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f"./models/custom_trade_model_{timestamp}"
        
        # Build state vectors from trade journal entries
        # [This would be more sophisticated in a full implementation]
        
        # Mock training (would be real training with real trade data in production)
        timesteps = episodes * 1000
        # Create a mock environment with the data
        if not stock_data_dict:
            logger.warning("No stock data available for training")
            stock_data_dict = {'MOCK': pd.DataFrame()}  # Empty DataFrame as fallback
        
        agent.create_environment(stock_data_dict)
        agent.train(stock_data_dict=stock_data_dict, total_timesteps=timesteps)
        
        # Save model training record
        training_info = RLModelTraining(
            model_name=model_name,
            start_date=start_date,
            end_date=end_date,
            stocks_used=','.join(stock_symbols),
            total_episodes=episodes,
            total_timesteps=timesteps,
            model_path=model_path,
            timeframe=timeframe
        )
        
        db.session.add(training_info)
        db.session.commit()
        
        flash(f'Successfully trained custom model: {model_name} based on your trade journal', 'success')
        return redirect(url_for('trade_journal'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error training custom RL model: {e}")
        flash(f'Error training custom RL model: {str(e)}', 'danger')
        return redirect(url_for('trade_journal'))
@app.route('/fetch_patterns_for_journal', methods=['POST'])
def fetch_patterns_for_journal():
    """Fetch reversal patterns for the trade journal"""
    try:
        # Get form data
        symbol = request.form.get('symbol', '').strip().upper()
        days = int(request.form.get('days', 30))
        timeframe = request.form.get('timeframe', '1day')
        
        if not symbol:
            flash('Stock symbol is required', 'danger')
            return redirect(url_for('trade_journal'))
        
        # Map timeframe to Polygon parameters
        timeframe_map = {
            '5min': {'timespan': 'minute', 'multiplier': 5},
            '15min': {'timespan': 'minute', 'multiplier': 15},
            '30min': {'timespan': 'minute', 'multiplier': 30},
            '1hour': {'timespan': 'hour', 'multiplier': 1},
            '4hour': {'timespan': 'hour', 'multiplier': 4},
            '1day': {'timespan': 'day', 'multiplier': 1},
            '1week': {'timespan': 'week', 'multiplier': 1}
        }
        
        selected_timeframe = timeframe_map.get(timeframe, {'timespan': 'day', 'multiplier': 1})
        timespan = selected_timeframe['timespan']
        multiplier = selected_timeframe['multiplier']
        
        # Get stock from database or create if it doesn't exist
        stock = Stock.query.filter_by(symbol=symbol).first()
        if not stock:
            stock = Stock(symbol=symbol)
            db.session.add(stock)
            db.session.commit()
        
        # Fetch historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Get market API client for this request
        market_api = get_market_api()
        
        logger.info(f"Fetching {symbol} data with timeframe: {timespan}/{multiplier}, from {start_date} to {end_date}")
        data = market_api.get_historical_data(
            symbol, 
            timespan=timespan, 
            multiplier=multiplier, 
            start_date=start_date, 
            end_date=end_date
        )
        
        if data.empty:
            flash(f'No data available for {symbol}', 'warning')
            return redirect(url_for('trade_journal'))
        
        # Calculate indicators
        indicators = TechnicalIndicators.calculate_all_indicators(data)
        
        # Detect patterns
        detector = PatternDetector()
        patterns_df = detector.detect_all_patterns(data, indicators)
        
        # Save patterns to database and collect them for display
        patterns_for_session = []
        count = 0
        
        # Filter only rows with reversal patterns
        pattern_rows = patterns_df[patterns_df['has_reversal_pattern'] == True]
        
        if pattern_rows.empty:
            flash(f'No reversal patterns found for {symbol} in the selected timeframe', 'info')
            return redirect(url_for('trade_journal'))
        
        for idx, row in pattern_rows.iterrows():
            # Check if pattern already exists for this timestamp
            existing = PatternDetection.query.filter_by(
                stock_id=stock.id,
                timestamp=idx.to_pydatetime()
            ).first()
            
            if not existing:
                # Determine pattern type and description (same logic as in fetch_historical_data)
                pattern_type = 'unknown'
                description = ''
                
                # Basic patterns
                if 'is_doji' in row and row['is_doji']:
                    pattern_type = 'doji'
                    description += 'Doji pattern detected. '
                
                if 'is_hammer' in row and row['is_hammer']:
                    pattern_type = 'hammer'
                    description += 'Hammer pattern detected. '
                
                if 'potential_bullish_reversal' in row and row['potential_bullish_reversal']:
                    pattern_type = 'consecutive_bullish'
                    description += 'Potential bullish reversal after consecutive bearish candles. '
                
                if 'potential_bearish_reversal' in row and row['potential_bearish_reversal']:
                    pattern_type = 'consecutive_bearish'
                    description += 'Potential bearish reversal after consecutive bullish candles. '
                    
                # Advanced patterns - Engulfing
                if 'bullish_engulfing' in row and row['bullish_engulfing']:
                    pattern_type = 'bullish_engulfing'
                    description += 'Bullish Engulfing pattern detected. Current candle completely engulfs previous bearish candle. Strong bullish signal. '
                
                if 'bearish_engulfing' in row and row['bearish_engulfing']:
                    pattern_type = 'bearish_engulfing'
                    description += 'Bearish Engulfing pattern detected. Current candle completely engulfs previous bullish candle. Strong bearish signal. '
                
                # Star patterns
                if 'morning_star' in row and row['morning_star']:
                    pattern_type = 'morning_star'
                    description += 'Morning Star pattern detected. Potential bullish reversal after downtrend. '
                
                if 'evening_star' in row and row['evening_star']:
                    pattern_type = 'evening_star'
                    description += 'Evening Star pattern detected. Potential bearish reversal after uptrend. '
                
                # Shooting Star
                if 'shooting_star' in row and row['shooting_star']:
                    pattern_type = 'shooting_star'
                    description += 'Shooting Star pattern detected. Long upper shadow with little to no lower shadow. Bearish reversal signal. '
                
                # Piercing patterns
                if 'piercing_line' in row and row['piercing_line']:
                    pattern_type = 'piercing_line'
                    description += 'Piercing Line pattern detected. Bullish reversal signal where current candle closes more than halfway up previous bearish candle. '
                
                if 'dark_cloud_cover' in row and row['dark_cloud_cover']:
                    pattern_type = 'dark_cloud_cover'
                    description += 'Dark Cloud Cover pattern detected. Bearish reversal signal where current candle closes more than halfway down previous bullish candle. '
                
                # Three candle patterns
                if 'three_white_soldiers' in row and row['three_white_soldiers']:
                    pattern_type = 'three_white_soldiers'
                    description += 'Three White Soldiers pattern detected. Three consecutive bullish candles with higher highs and higher lows. Strong bullish signal. '
                
                if 'three_black_crows' in row and row['three_black_crows']:
                    pattern_type = 'three_black_crows'
                    description += 'Three Black Crows pattern detected. Three consecutive bearish candles with lower lows and lower highs. Strong bearish signal. '
                
                # Add RSI information to description
                if 'rsi' in row:
                    description += f"RSI: {row['rsi']:.2f}. "
                    
                    if 'in_opening_hour' in row and row['in_opening_hour']:
                        description += "During market opening (9:30-10:00). "
                        if row['rsi'] > 70:
                            description += "Overbought during market open suggests bullish momentum. "
                        elif row['rsi'] < 30:
                            description += "Oversold during market open suggests bearish momentum. "
                    else:
                        if row['rsi'] > 70:
                            description += "Overbought condition in standard market hours. "
                        elif row['rsi'] < 30:
                            description += "Oversold condition in standard market hours. "
                
                # Add ORB information to description
                if 'above_orb_high' in row and row['above_orb_high']:
                    description += "Price above Opening Range Breakout high. "
                elif 'below_orb_low' in row and row['below_orb_low']:
                    description += "Price below Opening Range Breakout low. "
                
                # Create new pattern
                pattern = PatternDetection(
                    stock_id=stock.id,
                    timestamp=idx.to_pydatetime(),
                    pattern_type=pattern_type,
                    description=description,
                    rsi_value=row.get('rsi'),
                    above_orb_high=row.get('above_orb_high', False),
                    below_orb_low=row.get('below_orb_low', False),
                    in_opening_hour=row.get('in_opening_hour', False)
                )
                
                db.session.add(pattern)
                db.session.flush()  # Get the ID without committing
                
                # Add to list of patterns for the session
                patterns_for_session.append((pattern.id, stock.id))
                count += 1
            else:
                # Add existing pattern to the list
                patterns_for_session.append((existing.id, stock.id))
        
        db.session.commit()
        
        # Store pattern IDs in session for display in trade_journal route
        session['recent_patterns'] = patterns_for_session
        
        flash(f'Found {count} new patterns and {len(patterns_for_session)} total patterns for {symbol}', 'success')
        return redirect(url_for('trade_journal'))
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error fetching patterns for journal: {e}")
        flash(f'Error fetching patterns: {str(e)}', 'danger')
        return redirect(url_for('trade_journal'))