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