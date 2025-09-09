import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration constants
TEAM_ID = 5409265
CURRENT_GW = 4
LOCKED_PLAYERS = ['Marc Guiu Paz']
LOOK_AHEAD_GWS = 6
MIN_MINUTES_THRESHOLD = 45
TRAINING_SAMPLE_SIZE = 500

# Captaincy scoring weights
CAPTAINCY_WEIGHTS = {
    'predicted_points': 0.35,
    'fixture_favorability': 0.25,
    'form_trend': 0.20,
    'consistency': 0.15,
    'home_advantage': 0.05
}

# Transfer strategy parameters
TRANSFER_STRATEGY = {
    'min_score_improvement': 8.0,
    'max_transfers_per_gw': 2,
    'price_change_weight': 0.1,
    'ownership_differential_threshold': 15.0
}

# Head-to-head prioritisation weights
H2H_WEIGHTS = {
    'attacking_player': 1.3,
    'home_advantage': 1.15,
    'form_multiplier': 1.2,
    'fixture_strength': 1.1,
    'predicted_points': 0.4,
    'consistency': 0.3
}

# Valid FPL formations
VALID_FORMATIONS = [
    {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
    {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},
    {'GK': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},
    {'GK': 1, 'DEF': 5, 'MID': 4, 'FWD': 1}
]

# Chip usage recommendations
CHIP_RECOMMENDATIONS = {
    'First Half': {
        'Wildcard': {
            'best_timing': 'GW 8-12',
            'reason': 'After international break, injury clarity, and fixture swing analysis'
        },
        'Bench Boost': {
            'best_timing': 'GW 6-10 (Double Gameweek)',
            'reason': 'Use when you have strong bench players with good fixtures'
        },
        'Triple Captain': {
            'best_timing': 'GW 4-8 (Double Gameweek)',
            'reason': 'Best captaincy option with guaranteed two games'
        },
        'Assistant Manager': {
            'best_timing': 'GW 12-16',
            'reason': 'Before difficult fixture periods or injury crises'
        }
    },
    'Second Half': {
        'Wildcard': {
            'best_timing': 'GW 20-24',
            'reason': 'Mid-season reset, target blank/double gameweeks'
        },
        'Bench Boost': {
            'best_timing': 'GW 25-29 (Double Gameweek)',
            'reason': 'Maximum games when combined with doubles'
        },
        'Triple Captain': {
            'best_timing': 'GW 30-34 (Double Gameweek)',
            'reason': 'Final run-in with premium captains having doubles'
        },
        'Assistant Manager': {
            'best_timing': 'GW 35-38',
            'reason': 'Final stretch optimisation and rotation management'
        }
    }
}

# API access functions
def get_bootstrap(): 
    return requests.get('https://fantasy.premierleague.com/api/bootstrap-static/').json()

def get_player_history(pid): 
    return requests.get(f'https://fantasy.premierleague.com/api/element-summary/{pid}/').json()

def get_team_picks(team_id, gw): 
    return requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/event/{gw}/picks/').json()

def get_team_info(team_id): 
    return requests.get(f'https://fantasy.premierleague.com/api/entry/{team_id}/').json()

def get_fixtures(): 
    return pd.DataFrame(requests.get('https://fantasy.premierleague.com/api/fixtures/').json())

def get_current_gameweek_data():
    '''Get current gameweek information'''
    bootstrap = get_bootstrap()
    current_gw = None
    for event in bootstrap['events']:
        if event['is_current']:
            current_gw = event['id']
            break
    return current_gw if current_gw else CURRENT_GW

def get_last_gw_points(player_id):
    '''Get points from the most recent completed gameweek'''
    try:
        history = get_player_history(player_id)
        if history and 'history' in history and history['history']:
            return history['history'][-1]['total_points']
        return 0
    except:
        return 0

# Utility functions
def clean_table_display(df, title='', max_rows=20):
    '''Display clean, formatted tables'''
    print(f'\n{"=" * 90}')
    print(f'{title.center(90)}')
    print(f'{"=" * 90}')
    
    if df.empty:
        print('No data available')
        return
    
    # Format numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df_display = df.copy()
    
    for col in numeric_cols:
        if col in ['cost', 'now_cost']:
            df_display[col] = df_display[col].apply(lambda x: f'£{x:.1f}m')
        elif 'percentage' in col or 'percent' in col or col == 'selected_by_percent' or col == 'ownership':
            df_display[col] = df_display[col].apply(lambda x: f'{x:.1f}%')
        elif col in ['predicted_points', 'form', 'xG', 'xA', 'xGI', 'fixture_favorability', 'fixture_rating', 'h2h_priority_score', 'attacking_favorability', 'defensive_favorability']:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.2f}')
        elif col in ['score', 'comprehensive_score']:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.1f}')
        elif col in ['chance_of_playing_this_round', 'chance_of_playing_next_round']:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.0f}%' if pd.notna(x) else '100%')
        elif col == 'last_gw_points':
            df_display[col] = df_display[col].apply(lambda x: f'{x:.0f}')
        else:
            df_display[col] = df_display[col].apply(lambda x: f'{x:.0f}' if pd.notna(x) else '0')
    
    # Truncate long names
    if 'full_name' in df_display.columns:
        df_display['full_name'] = df_display['full_name'].str[:20]
    if 'team_name' in df_display.columns:
        df_display['team_name'] = df_display['team_name'].str[:12]
    if 'opponent_team' in df_display.columns:
        df_display['opponent_team'] = df_display['opponent_team'].str[:12]
    
    print(df_display.head(max_rows).to_string(index=False, max_colwidth=15))
    print(f'{"=" * 90}\n')

def display_chip_strategy():
    '''Display chip usage strategy for 2025/26 season'''
    print('\n' + '=' * 90)
    print('CHIP STRATEGY GUIDE FOR 2025/26 SEASON'.center(90))
    print('=' * 90)
    
    for half, chips in CHIP_RECOMMENDATIONS.items():
        print(f'\n{half.upper()} SEASON CHIPS:')
        print('-' * 50)
        
        for chip_name, strategy in chips.items():
            print(f'\n{chip_name.upper()}:')
            print(f'  Best Timing: {strategy["best_timing"]}')
            print(f'  Strategy: {strategy["reason"]}')
        
        if half == 'First Half':
            print(f'\nFirst Half Priority Order: Triple Captain → Bench Boost → Wildcard → Assistant Manager')
        else:
            print(f'\nSecond Half Priority Order: Wildcard → Bench Boost → Triple Captain → Assistant Manager')

# Main analysis class
class AdvancedFPLAnalyser:
    def __init__(self):
        '''Initialise the FPL analyser with comprehensive data'''
        print('Initialising Advanced FPL Analyser...')
        
        # Fetch base data from API
        self.bootstrap = get_bootstrap()
        self.players = pd.DataFrame(self.bootstrap['elements'])
        self.teams = pd.DataFrame(self.bootstrap['teams'])
        self.positions = pd.DataFrame(self.bootstrap['element_types'])
        self.fixtures = get_fixtures()
        
        # Setup player data structure
        self._setup_player_data()
        
        # Process availability and injury data
        self._process_availability_data()
        
        # Calculate comprehensive statistics
        self._calculate_comprehensive_stats()
        
        print('FPL Analyser initialisation complete!')

    def _setup_player_data(self):
        '''Setup merged player data with teams and positions'''
        # Merge team information
        self.players['full_name'] = self.players['first_name'] + ' ' + self.players['second_name']
        self.players = self.players.merge(
            self.teams[['id', 'name']], 
            left_on='team', 
            right_on='id', 
            suffixes=('', '_team')
        )
        self.players.rename(columns={'name': 'team_name'}, inplace=True)
        
        # Merge position information
        self.players = self.players.merge(
            self.positions[['id', 'singular_name']], 
            left_on='element_type', 
            right_on='id', 
            suffixes=('', '_pos')
        )
        self.players.rename(columns={'singular_name': 'position'}, inplace=True)
        
        # Convert cost to actual value
        self.players['cost'] = self.players['now_cost'] / 10.0
        
        # Get last gameweek points
        print('Fetching last gameweek points...')
        self.players['last_gw_points'] = self.players['id'].apply(get_last_gw_points)

    def _process_availability_data(self):
        '''Process injury and suspension data with live API updates'''
        print('Processing current injury and availability data...')
        
        # Fill missing availability data with default values
        self.players['chance_of_playing_this_round'] = self.players['chance_of_playing_this_round'].fillna(100)
        self.players['chance_of_playing_next_round'] = self.players['chance_of_playing_next_round'].fillna(100)
        
        # Create availability flags
        self.players['is_available'] = self.players['chance_of_playing_this_round'] > 0
        self.players['is_likely_available'] = self.players['chance_of_playing_this_round'] >= 75
        self.players['is_doubtful'] = (self.players['chance_of_playing_this_round'] > 0) & (self.players['chance_of_playing_this_round'] < 75)
        self.players['is_injured'] = self.players['chance_of_playing_this_round'] == 0
        
        # Process injury news
        self.players['has_injury_news'] = self.players['news'].notna() & (self.players['news'] != '')
        self.players['injury_status'] = self.players.apply(self._determine_injury_status, axis=1)
        
        # Calculate availability multiplier for scoring
        self.players['availability_multiplier'] = self.players['chance_of_playing_this_round'] / 100.0
        
        # Display availability summary
        print(f'Live availability summary:')
        print(f'  Available players: {self.players["is_available"].sum()}')
        print(f'  Likely available: {self.players["is_likely_available"].sum()}')
        print(f'  Doubtful: {self.players["is_doubtful"].sum()}')
        print(f'  Injured/Suspended: {self.players["is_injured"].sum()}')
        
        # Show key injury updates
        injured_or_doubtful = self.players[
            (self.players['chance_of_playing_this_round'] < 100) & 
            (self.players['total_points'] > 20)
        ]
        
        if not injured_or_doubtful.empty:
            print(f'\nKey injury updates:')
            for _, player in injured_or_doubtful.iterrows():
                print(f'  {player["full_name"]}: {player["chance_of_playing_this_round"]:.0f}% chance')

    def _determine_injury_status(self, row):
        '''Determine injury status from live API data'''
        chance = row['chance_of_playing_this_round']
        if chance == 0:
            return 'Unavailable'
        elif chance == 25:
            return 'Major Doubt'
        elif chance == 50:
            return 'Doubt'  
        elif chance == 75:
            return 'Minor Doubt'
        else:
            return 'Available'

    def _safe_convert(self, x, dtype='float'):
        '''Safely convert values with error handling'''
        if pd.isna(x) or x == '':
            return 0.0 if dtype == 'float' else 0
        try:
            return float(x) if dtype == 'float' else int(float(x))
        except:
            return 0.0 if dtype == 'float' else 0

    def _calculate_comprehensive_stats(self):
        '''Calculate comprehensive performance metrics using all FPL stats'''
        print('Calculating comprehensive statistics using all available FPL data...')
        
        # List of all statistical columns from FPL API
        all_stat_columns = [
            'goals_scored', 'assists', 'total_points', 'minutes', 'goals_conceded', 
            'creativity', 'influence', 'threat', 'bonus', 'bps', 'ict_index', 
            'clean_sheets', 'saves', 'penalties_saved', 'penalties_missed', 
            'yellow_cards', 'red_cards', 'own_goals', 'penalties_order',
            'direct_freekicks_order', 'corners_and_indirect_freekicks_order',
            'expected_goals', 'expected_assists', 'expected_goal_involvements', 
            'expected_goals_conceded', 'value_form', 'value_season', 
            'points_per_game', 'transfers_in_event', 'transfers_out_event',
            'selected_by_percent', 'dreamteam_count', 'in_dreamteam',
            'form', 'special'
        ]
        
        # Process all available columns with safe conversion
        for col in all_stat_columns:
            if col in self.players.columns:
                self.players[col] = self.players[col].apply(lambda x: self._safe_convert(x, 'float'))
        
        # Create derived statistics
        self.players['xGI'] = self.players['expected_goals'] + self.players['expected_assists']
        self.players['goal_contributions'] = self.players['goals_scored'] + self.players['assists']
        self.players['defensive_actions'] = 0  # API doesn't provide detailed defensive stats
        self.players['games_played'] = (self.players['minutes'] / 90).round()
        self.players['points_per_90'] = np.where(
            self.players['minutes'] > 0,
            (self.players['total_points'] / self.players['minutes']) * 90,
            0
        )
        
        # Calculate form metrics
        self._calculate_form_metrics()
        
        print('Comprehensive statistics calculated successfully!')

    def _calculate_form_metrics(self):
        '''Calculate form, consistency, and momentum metrics'''
        print('Calculating form metrics for available players...')
        
        # Initialise default values
        self.players['form_3'] = 0
        self.players['form_5'] = 0
        self.players['form_10'] = 0
        self.players['consistency'] = 0.5
        self.players['ceiling'] = 0
        self.players['floor'] = 0
        self.players['momentum'] = 0
        self.players['trend'] = 'stable'
        
        # Sample players for detailed analysis
        available_players = self.players[self.players['is_available']]['id'].tolist()
        sample_size = min(TRAINING_SAMPLE_SIZE or len(available_players), len(available_players))
        
        if sample_size > 0:
            sampled_players = np.random.choice(available_players, size=sample_size, replace=False).tolist()
            
            print(f'Analysing detailed form data for {len(sampled_players)} players...')
            
            for i, pid in enumerate(sampled_players):
                if i % 30 == 0:
                    print(f'  Processing player {i+1}/{len(sampled_players)}...')
                
                try:
                    player_data = get_player_history(pid)
                    if not player_data or 'history' not in player_data:
                        continue
                    
                    hist = player_data['history']
                    if not hist or len(hist) < 3:
                        continue
                    
                    # Extract point and minute data
                    points = [h['total_points'] for h in hist]
                    minutes = [h['minutes'] for h in hist]
                    
                    # Filter for meaningful games
                    recent_performances = [(p, m) for p, m in zip(points, minutes) if m >= MIN_MINUTES_THRESHOLD]
                    if not recent_performances:
                        recent_performances = list(zip(points, minutes))
                    
                    recent_points = [p for p, m in recent_performances]
                    
                    # Calculate form metrics
                    form_3 = np.mean(recent_points[-3:]) if len(recent_points) >= 3 else np.mean(recent_points)
                    form_5 = np.mean(recent_points[-5:]) if len(recent_points) >= 5 else np.mean(recent_points)
                    form_10 = np.mean(recent_points[-10:]) if len(recent_points) >= 10 else np.mean(recent_points)
                    
                    # Update player data
                    player_mask = self.players['id'] == pid
                    self.players.loc[player_mask, 'form_3'] = form_3
                    self.players.loc[player_mask, 'form_5'] = form_5
                    self.players.loc[player_mask, 'form_10'] = form_10
                    
                    # Calculate consistency metrics
                    if len(recent_points) >= 5:
                        consistency = 1 - (np.std(recent_points) / (np.mean(recent_points) + 1))
                        ceiling = np.percentile(recent_points, 80)
                        floor = np.percentile(recent_points, 20)
                    else:
                        consistency = 0.5
                        ceiling = max(recent_points) if recent_points else 0
                        floor = min(recent_points) if recent_points else 0
                    
                    self.players.loc[player_mask, 'consistency'] = max(0, consistency)
                    self.players.loc[player_mask, 'ceiling'] = ceiling
                    self.players.loc[player_mask, 'floor'] = floor
                    
                    # Calculate momentum and trend
                    if len(recent_points) >= 6:
                        recent_trend = np.mean(recent_points[-3:]) - np.mean(recent_points[-6:-3])
                        momentum = recent_trend / (np.mean(recent_points[-6:]) + 1)
                        
                        if momentum > 0.2:
                            trend = 'rising'
                        elif momentum < -0.2:
                            trend = 'falling'
                        else:
                            trend = 'stable'
                    else:
                        momentum = 0
                        trend = 'stable'
                    
                    self.players.loc[player_mask, 'momentum'] = momentum
                    self.players.loc[player_mask, 'trend'] = trend
                    
                except Exception as e:
                    continue
        
        print('Form metrics calculation complete!')

    def calculate_fixture_analysis(self):
        '''Calculate fixture difficulty with team strength analysis'''
        print('Analysing fixture difficulty with team strength analysis...')
        
        # Calculate team strength metrics
        team_stats = {}
        current_gw = CURRENT_GW
        
        for team_id in self.teams['id']:
            team_players = self.players[self.players['team'] == team_id]
            
            # Calculate team metrics
            total_goals_scored = team_players['goals_scored'].sum()
            total_goals_conceded = team_players['goals_conceded'].sum()
            total_clean_sheets = team_players[team_players['position'] == 'Goalkeeper']['clean_sheets'].sum()
            
            # Estimate games played
            estimated_games = max(1, current_gw - 1)
            
            # Team strength metrics
            attack_strength = total_goals_scored / estimated_games
            defense_strength = total_goals_conceded / estimated_games
            clean_sheet_rate = total_clean_sheets / estimated_games
            
            team_stats[team_id] = {
                'attack_strength': attack_strength,
                'defense_strength': defense_strength,
                'clean_sheet_rate': clean_sheet_rate
            }
        
        # Apply fixture metrics to all players
        fixture_data = []
        for team_id in self.players['team'].unique():
            metrics = self._get_fixture_metrics(team_id, team_stats)
            metrics['team_id'] = team_id
            fixture_data.append(metrics)
        
        fixture_df = pd.DataFrame(fixture_data)
        self.players = self.players.merge(
            fixture_df, left_on='team', right_on='team_id', how='left'
        ).drop('team_id', axis=1)
        
        print('Fixture analysis complete!')

    def _get_fixture_metrics(self, team_id, team_stats):
        '''Calculate detailed fixture metrics for a team'''
        # Get upcoming fixtures
        upcoming = self.fixtures[
            (self.fixtures['event'] > CURRENT_GW - 1) & 
            (self.fixtures['event'] <= CURRENT_GW + LOOK_AHEAD_GWS)
        ]
        
        team_fixtures = upcoming[
            (upcoming['team_h'] == team_id) | (upcoming['team_a'] == team_id)
        ]
        
        if team_fixtures.empty:
            return self._get_default_fixture_metrics()
        
        difficulties = []
        attacking_favorability = []
        defensive_favorability = []
        home_count = 0
        away_count = 0
        
        for _, fixture in team_fixtures.iterrows():
            is_home = fixture['team_h'] == team_id
            opponent_id = fixture['team_a'] if is_home else fixture['team_h']
            
            # Standard difficulty from FPL API
            if is_home:
                difficulty = fixture.get('team_h_difficulty', 3)
                home_count += 1
            else:
                difficulty = fixture.get('team_a_difficulty', 3)
                away_count += 1
            
            difficulties.append(difficulty)
            
            # Calculate attacking and defensive favorability
            if opponent_id in team_stats:
                opponent = team_stats[opponent_id]
                league_avg = 1.4
                
                # Attacking favorability based on opponent's defence
                opponent_def_strength = opponent['defense_strength']
                if opponent_def_strength > league_avg:
                    attack_favor = min(1.0, 0.5 + (opponent_def_strength - league_avg) * 0.4)
                else:
                    attack_favor = max(0.0, 0.5 - (league_avg - opponent_def_strength) * 0.4)
                
                # Defensive favorability based on opponent's attack
                opponent_att_strength = opponent['attack_strength']
                if opponent_att_strength < league_avg:
                    def_favor = min(1.0, 0.5 + (league_avg - opponent_att_strength) * 0.4)
                else:
                    def_favor = max(0.0, 0.5 - (opponent_att_strength - league_avg) * 0.4)
                
                # Apply home advantage
                if is_home:
                    attack_favor = min(1.0, attack_favor * 1.15)
                    def_favor = min(1.0, def_favor * 1.1)
                
                attacking_favorability.append(attack_favor)
                defensive_favorability.append(def_favor)
            else:
                attacking_favorability.append(0.5)
                defensive_favorability.append(0.5)
        
        # Calculate final metrics
        avg_difficulty = np.mean(difficulties) if difficulties else 3.0
        fixture_favorability = (6 - avg_difficulty) / 5
        
        avg_attacking_favorability = np.mean(attacking_favorability) if attacking_favorability else 0.5
        avg_defensive_favorability = np.mean(defensive_favorability) if defensive_favorability else 0.5
        
        return {
            'avg_difficulty': avg_difficulty,
            'home_fixtures': home_count,
            'away_fixtures': away_count,
            'double_gws': 0,
            'blank_gws': 0,
            'fixture_favorability': fixture_favorability,
            'fixture_rating': fixture_favorability,
            'attacking_favorability': avg_attacking_favorability,
            'defensive_favorability': avg_defensive_favorability,
            'next_5_avg_difficulty': avg_difficulty
        }

    def _get_default_fixture_metrics(self):
        '''Return default fixture metrics when no data available'''
        return {
            'avg_difficulty': 3.0,
            'home_fixtures': 0,
            'away_fixtures': 0,
            'double_gws': 0,
            'blank_gws': 0,
            'fixture_favorability': 0.4,
            'fixture_rating': 0.4,
            'attacking_favorability': 0.5,
            'defensive_favorability': 0.5,
            'gw1_difficulty': 3.0,
            'gw2_difficulty': 3.0,
            'gw3_difficulty': 3.0,
            'gw4_difficulty': 3.0,
            'gw5_difficulty': 3.0,
            'next_5_avg_difficulty': 3.0
        }

    def build_comprehensive_ml_model(self):
        '''Build ML model with realistic target values'''
        print('Training comprehensive ML model with realistic bounds...')
        
        training_rows = []
        available_players = self.players[
            (self.players['is_available']) & 
            (self.players['minutes'] > 0)
        ]
        
        # Select players for training
        if len(available_players) > 400:
            priority_players = available_players.nlargest(400, ['total_points', 'minutes'])
            training_player_ids = priority_players['id'].tolist()
        else:
            training_player_ids = available_players['id'].tolist()
        
        print(f'Training on {len(training_player_ids)} players with comprehensive stats...')
        
        successful_requests = 0
        for i, pid in enumerate(training_player_ids):
            try:
                if i % 25 == 0:
                    print(f'   Processing player {i+1}/{len(training_player_ids)}...')
                
                player_data = get_player_history(pid)
                
                if not isinstance(player_data, dict) or 'history' not in player_data:
                    continue
                    
                hist = player_data['history']
                if not hist:
                    continue
                
                player_info = self.players.loc[self.players['id'] == pid].iloc[0]
                successful_requests += 1
                
                # Create training samples
                for j, h in enumerate(hist):
                    if h['minutes'] < 10:
                        continue
                    
                    recent_games = hist[max(0, j-3):j] if j > 0 else []
                    points_trend = 0
                    if recent_games and len(recent_games) >= 2:
                        recent_points = [g['total_points'] for g in recent_games]
                        points_trend = recent_points[-1] - np.mean(recent_points[:-1])
                    
                    # Comprehensive feature set
                    training_row = {
                        'player_id': pid,
                        'position': player_info['position'],
                        'minutes': min(90, h['minutes']),
                        'goals_scored': h['goals_scored'],
                        'assists': h['assists'],
                        'total_points': h['total_points'],
                        'clean_sheets': h['clean_sheets'],
                        'goals_conceded': h['goals_conceded'],
                        'own_goals': h.get('own_goals', 0),
                        'penalties_saved': h.get('penalties_saved', 0),
                        'penalties_missed': h.get('penalties_missed', 0),
                        'yellow_cards': h.get('yellow_cards', 0),
                        'red_cards': h.get('red_cards', 0),
                        'saves': h.get('saves', 0),
                        'bonus': h['bonus'],
                        'bps': self._safe_convert(h.get('bps', 0), 'int'),
                        'influence': min(200, self._safe_convert(h.get('influence', 0))),
                        'creativity': min(200, self._safe_convert(h.get('creativity', 0))),
                        'threat': min(200, self._safe_convert(h.get('threat', 0))),
                        'ict_index': min(25, self._safe_convert(h.get('ict_index', 0))),
                        'expected_goals': min(2.0, self._safe_convert(h.get('expected_goals', 0))),
                        'expected_assists': min(2.0, self._safe_convert(h.get('expected_assists', 0))),
                        'expected_goal_involvements': min(3.0, self._safe_convert(h.get('expected_goal_involvements', 0))),
                        'expected_goals_conceded': min(5.0, self._safe_convert(h.get('expected_goals_conceded', 0))),
                        'was_home': h['was_home'],
                        'kickoff_time': h.get('kickoff_time', ''),
                        'team_h_score': min(10, h.get('team_h_score', 0)),
                        'team_a_score': min(10, h.get('team_a_score', 0)),
                        'round': h.get('round', 0),
                        'points_trend': max(-10, min(10, points_trend)),
                        'cost': player_info['cost'],
                        'availability': player_info['availability_multiplier'],
                        'total_season_points': player_info['total_points'],
                        'form': player_info.get('form', 0),
                        'dreamteam_count': player_info.get('dreamteam_count', 0),
                        'value_season': player_info.get('value_season', 0),
                        'selected_by_percent': player_info.get('selected_by_percent', 0),
                        'opponent_difficulty': h.get('opponent_team_difficulty', 3.0),
                        'fixture_favorability': player_info.get('fixture_favorability', 0.5),
                        'attacking_favorability': player_info.get('attacking_favorability', 0.5),
                        'defensive_favorability': player_info.get('defensive_favorability', 0.5),
                        'next_5_avg_difficulty': player_info.get('next_5_avg_difficulty', 3.0),
                        'points': min(25, max(0, h['total_points']))
                    }
                    
                    training_rows.append(training_row)
                    
            except Exception as e:
                continue
        
        print(f'Successfully processed {successful_requests} players')
        
        if not training_rows or len(training_rows) < 100:
            print('Insufficient training data - using enhanced fallback model')
            return self._create_comprehensive_fallback_model()
        
        training_df = pd.DataFrame(training_rows)
        print(f'Training dataset: {len(training_df)} samples with comprehensive FPL statistics')
        
        print(f'Training points distribution - Min: {training_df["points"].min():.1f}, Max: {training_df["points"].max():.1f}, Mean: {training_df["points"].mean():.2f}')
        
        # Feature engineering
        training_df['goals_assists'] = training_df['goals_scored'] + training_df['assists']
        training_df['xGI'] = training_df['expected_goals'] + training_df['expected_assists']
        training_df['minutes_factor'] = training_df['minutes'] / 90.0
        training_df['defensive_value'] = training_df['clean_sheets'] + training_df['saves'] * 0.1
        training_df['attacking_threat'] = training_df['goals_scored'] + training_df['assists'] * 0.5
        training_df['cards_penalty'] = training_df['yellow_cards'] * (-0.5) + training_df['red_cards'] * (-1.5)
        
        # Position encoding
        for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            training_df[f'is_{pos.lower()}'] = (training_df['position'] == pos).astype(int)
        
        # Feature set
        feature_columns = [
            'minutes', 'goals_scored', 'assists', 'clean_sheets', 'goals_conceded', 
            'own_goals', 'penalties_saved', 'penalties_missed', 'yellow_cards', 'red_cards', 'saves',
            'bonus', 'bps', 'influence', 'creativity', 'threat', 'ict_index', 
            'expected_goals', 'expected_assists', 'expected_goal_involvements', 'expected_goals_conceded',
            'was_home', 'team_h_score', 'team_a_score', 'points_trend',
            'cost', 'availability', 'total_season_points', 'form', 'dreamteam_count', 
            'value_season', 'selected_by_percent', 'opponent_difficulty', 'fixture_favorability', 
            'attacking_favorability', 'defensive_favorability', 'next_5_avg_difficulty',
            'goals_assists', 'xGI', 'minutes_factor', 'defensive_value', 
            'attacking_threat', 'cards_penalty', 'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward'
        ]
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['points']
        
        # Model selection based on data size
        if len(X) < 300:
            model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
        elif len(X) < 1500:
            model = GradientBoostingRegressor(
                n_estimators=300, max_depth=8, learning_rate=0.08, 
                random_state=42, subsample=0.8
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=400, max_depth=10, learning_rate=0.06, 
                random_state=42, subsample=0.75, max_features='sqrt'
            )
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        
        # Model evaluation
        test_score = model.score(X_test, y_test) if len(X_test) > 0 else 0
        predictions = model.predict(X_test) if len(X_test) > 0 else []
        mae = mean_absolute_error(y_test, predictions) if len(predictions) > 0 else 0
        
        print(f'Comprehensive model trained - R²: {test_score:.3f}, MAE: {mae:.3f}')
        print(f'Features: {len(feature_columns)} (all available FPL statistics)')
        
        # Feature importance analysis
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print('Top 15 most important features:')
            for _, row in feature_importance.head(15).iterrows():
                print(f'  {row["feature"]}: {row["importance"]:.3f}')
        
        return model, feature_columns

    def _create_comprehensive_fallback_model(self):
        '''Comprehensive fallback model using current season stats'''
        print('Creating comprehensive fallback prediction model...')
        
        feature_columns = [
            'minutes', 'goals_scored', 'assists', 'total_points', 'clean_sheets', 
            'goals_conceded', 'saves', 'penalties_saved', 'penalties_missed',
            'yellow_cards', 'red_cards', 'own_goals', 'bonus', 'bps',
            'influence', 'creativity', 'threat', 'ict_index', 
            'expected_goals', 'expected_assists', 'expected_goal_involvements',
            'expected_goals_conceded', 'dreamteam_count', 'form', 'value_season',
            'cost', 'points_per_game', 'availability_multiplier', 'selected_by_percent',
            'xGI', 'goal_contributions', 'defensive_actions', 'points_per_90',
            'fixture_favorability', 'attacking_favorability', 'defensive_favorability', 
            'next_5_avg_difficulty',
            'is_goalkeeper', 'is_defender', 'is_midfielder', 'is_forward'
        ]
        
        # Add position indicators
        for pos in ['Goalkeeper', 'Defender', 'Midfielder', 'Forward']:
            self.players[f'is_{pos.lower()}'] = (self.players['position'] == pos).astype(int)
        
        training_data = []
        available_players = self.players[
            (self.players['is_available']) & 
            (self.players['minutes'] > 90) & 
            (self.players['total_points'] > 0)
        ]
        
        # Generate synthetic training data with noise
        for _, player in available_players.iterrows():
            for variation in range(4):
                noise_factor = 1 + np.random.normal(0, 0.08)
                
                training_row = {}
                for col in feature_columns:
                    if col in player.index:
                        base_val = player[col] if pd.notna(player[col]) else 0
                        if col in ['minutes', 'influence', 'creativity', 'threat', 'ict_index']:
                            training_row[col] = base_val * noise_factor
                        else:
                            training_row[col] = base_val
                    else:
                        training_row[col] = 0
                
                base_ppg = player['total_points'] / max(1, player['minutes']/90)
                training_row['points'] = min(20, max(0, base_ppg * noise_factor))
                training_data.append(training_row)
        
        if len(training_data) < 20:
            print('Insufficient data for fallback model')
            return None, feature_columns
        
        training_df = pd.DataFrame(training_data)
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in training_df.columns:
                training_df[col] = 0
        
        X = training_df[feature_columns].fillna(0)
        y = training_df['points']
        
        model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X, y)
        
        print('Comprehensive fallback model created with realistic bounds')
        return model, feature_columns

    def generate_comprehensive_predictions(self, model, feature_columns):
        '''Generate realistic predictions with fixture adjustments'''
        if model is None:
            print('No model available for predictions')
            self.players['predicted_points'] = self.players['form']
            return
        
        print('Generating comprehensive predictions with realistic bounds...')
        
        pred_df = self.players.copy()
        
        # Feature engineering for predictions
        pred_df['goals_assists'] = pred_df['goals_scored'] + pred_df['assists']
        pred_df['xGI'] = pred_df['expected_goals'] + pred_df['expected_assists']
        pred_df['minutes_factor'] = pred_df['minutes'] / 90.0
        pred_df['defensive_value'] = pred_df['clean_sheets'] + pred_df.get('saves', 0) * 0.1
        pred_df['attacking_threat'] = pred_df['goals_scored'] + pred_df['assists'] * 0.5
        pred_df['cards_penalty'] = pred_df.get('yellow_cards', 0) * (-0.5) + pred_df.get('red_cards', 0) * (-1.5)
        
        # Context variables for predictions
        pred_df['was_home'] = 0.5
        pred_df['team_h_score'] = 1.5
        pred_df['team_a_score'] = 1.2
        pred_df['points_trend'] = pred_df.get('momentum', 0)
        pred_df['availability'] = pred_df['availability_multiplier']
        pred_df['opponent_difficulty'] = pred_df.get('next_5_avg_difficulty', 3.0)
        
        # Ensure all features exist
        for col in feature_columns:
            if col not in pred_df.columns:
                if 'difficulty' in col:
                    pred_df[col] = 3.0
                elif 'fixture' in col or 'favorability' in col:
                    pred_df[col] = 0.5
                elif 'is_' in col:
                    pred_df[col] = 0
                else:
                    pred_df[col] = 0
        
        X_pred = pred_df[feature_columns].fillna(0)
        raw_predictions = model.predict(X_pred)
        
        print(f'Raw predictions - Min: {raw_predictions.min():.2f}, Max: {raw_predictions.max():.2f}, Mean: {raw_predictions.mean():.2f}')
        
        # Apply availability adjustment
        availability_adjusted = raw_predictions * pred_df['availability_multiplier']
        
        # Calculate fixture adjustments
        fixture_multipliers = pred_df.apply(self._calculate_fixture_multiplier, axis=1)
        fixture_adjusted = availability_adjusted * fixture_multipliers
        
        # Apply regression to mean
        historical_avg = pred_df['points_per_game'].fillna(pred_df['form'].fillna(4))
        
        regression_strength = np.where(fixture_adjusted > 15, 0.6, 
                                      np.where(fixture_adjusted > 10, 0.4, 0.2))
        
        final_predictions = (1 - regression_strength) * fixture_adjusted + regression_strength * historical_avg
        
        # Apply realistic bounds
        final_predictions = np.clip(final_predictions, 0, 22)
        
        # Set injured players to 0
        final_predictions = np.where(pred_df['is_injured'], 0, final_predictions)
        
        pred_df['predicted_points'] = final_predictions
        
        print(f'Final predictions - Min: {final_predictions.min():.2f}, Max: {final_predictions.max():.2f}, Mean: {final_predictions.mean():.2f}')
        
        # Display high predictions for validation
        extreme_predictions = pred_df[pred_df['predicted_points'] > 15]
        if not extreme_predictions.empty:
            print(f'High predictions (>15): {len(extreme_predictions)} players')
            for _, player in extreme_predictions.head(3).iterrows():
                print(f'  {player["full_name"]}: {player["predicted_points"]:.1f} pts')
        
        self.players = pred_df
        print('Realistic predictions generated with fixture analysis')

    def _calculate_fixture_multiplier(self, row):
        '''Calculate fixture-based multiplier for predictions'''
        base_multiplier = 1.0
        
        # Difficulty adjustment
        difficulty_adj = (3.0 - row.get('next_5_avg_difficulty', 3.0)) * 0.02
        base_multiplier += difficulty_adj
        
        # Position-specific fixture bonuses
        if row['position'] == 'Forward':
            attacking_bonus = (row.get('attacking_favorability', 0.5) - 0.5) * 0.25
            base_multiplier += attacking_bonus
        elif row['position'] == 'Midfielder':
            attacking_bonus = (row.get('attacking_favorability', 0.5) - 0.5) * 0.15
            defensive_bonus = (row.get('defensive_favorability', 0.5) - 0.5) * 0.10
            base_multiplier += (attacking_bonus + defensive_bonus)
        elif row['position'] == 'Defender':
            defensive_bonus = (row.get('defensive_favorability', 0.5) - 0.5) * 0.20
            base_multiplier += defensive_bonus
        elif row['position'] == 'Goalkeeper':
            defensive_bonus = (row.get('defensive_favorability', 0.5) - 0.5) * 0.15
            if row.get('attacking_favorability', 0.5) < 0.3:
                defensive_bonus += 0.05
            base_multiplier += defensive_bonus
        
        return max(0.7, min(1.4, base_multiplier))

    def calculate_comprehensive_scores(self):
        '''Calculate comprehensive player scores using all available stats'''
        print('Calculating comprehensive scores with all FPL statistics...')
        
        df = self.players.copy()
        
        # Set injured players to 0 predicted points
        df.loc[df['is_injured'], 'predicted_points'] = 0
        
        # Normalise metrics for scoring
        metrics_to_normalise = [
            'predicted_points', 'form_5', 'consistency', 'fixture_favorability', 
            'attacking_favorability', 'defensive_favorability',
            'expected_goal_involvements', 'ict_index', 'threat', 'creativity', 
            'influence', 'bps', 'xGI', 'defensive_actions'
        ]
        
        for metric in metrics_to_normalise:
            if metric in df.columns:
                col_min = df[metric].min()
                col_max = df[metric].max()
                if col_max > col_min:
                    df[f'{metric}_norm'] = (df[metric] - col_min) / (col_max - col_min)
                else:
                    df[f'{metric}_norm'] = 0.5
        
        # Calculate comprehensive scores by position
        df['comprehensive_score'] = df.apply(self._calculate_position_score, axis=1)
        
        # Calculate value and differential scores
        df['value_score'] = (df['predicted_points'] * df['availability_multiplier']) / (df['cost'] + 0.1)
        
        df['ownership'] = df['selected_by_percent']
        df['differential_multiplier'] = np.where(
            df['ownership'] < TRANSFER_STRATEGY['ownership_differential_threshold'],
            1.25,
            1.0
        )
        df['differential_score'] = df['comprehensive_score'] * df['differential_multiplier']
        
        self.players = df

    def _calculate_position_score(self, row):
        '''Calculate position-specific comprehensive score'''
        pos = row['position']
        base_score = row.get('predicted_points_norm', 0) * 50
        
        # Apply availability penalty
        availability_penalty = 1.0
        if row['is_doubtful']:
            availability_penalty = 0.7
        elif row['is_injured']:
            availability_penalty = 0.0
        
        # Position-specific scoring with fixture bonuses
        if pos == 'Goalkeeper':
            score = base_score + (
                row['clean_sheets'] * 3 +
                row.get('saves', 0) * 0.5 +
                row.get('penalties_saved', 0) * 8 +
                row.get('fixture_favorability_norm', 0) * 25 +
                row.get('defensive_favorability_norm', 0) * 15 +
                row.get('consistency_norm', 0) * 20 +
                row.get('bps_norm', 0) * 15 +
                row.get('threat_norm', 0) * 10
            )
        elif pos == 'Defender':
            score = base_score + (
                row['clean_sheets'] * 4 +
                row.get('defensive_actions', 0) * 0.5 +
                row['goals_scored'] * 6 +
                row['assists'] * 3 +
                row.get('fixture_favorability_norm', 0) * 25 +
                row.get('defensive_favorability_norm', 0) * 20 +
                row.get('attacking_favorability_norm', 0) * 10 +
                row.get('consistency_norm', 0) * 15 +
                row.get('bps_norm', 0) * 15 +
                row.get('influence_norm', 0) * 10
            )
        elif pos == 'Midfielder':
            score = base_score + (
                row['goals_scored'] * 5 +
                row['assists'] * 3 +
                row.get('expected_goals', 0) * 3 +
                row.get('expected_assists', 0) * 3 +
                row.get('xGI_norm', 0) * 20 +
                row.get('threat_norm', 0) * 15 +
                row.get('creativity_norm', 0) * 15 +
                row.get('ict_index_norm', 0) * 20 +
                row.get('fixture_favorability_norm', 0) * 25 +
                row.get('attacking_favorability_norm', 0) * 15 +
                row.get('defensive_favorability_norm', 0) * 8 +
                row.get('form_5_norm', 0) * 18 +
                row.get('consistency_norm', 0) * 12
            )
        else:  # Forward
            score = base_score + (
                row['goals_scored'] * 6 +
                row['assists'] * 2 +
                row.get('expected_goals', 0) * 4 +
                row.get('threat_norm', 0) * 20 +
                row.get('ict_index_norm', 0) * 15 +
                row.get('form_5_norm', 0) * 20 +
                row.get('fixture_favorability_norm', 0) * 25 +
                row.get('attacking_favorability_norm', 0) * 20 +
                row.get('consistency_norm', 0) * 12 +
                row.get('bps_norm', 0) * 10
            )
        
        return max(0, score * availability_penalty)

    def get_current_team(self):
        '''Get the most recent team selection'''
        try:
            team_picks = get_team_picks(TEAM_ID, CURRENT_GW)
            if 'picks' not in team_picks:
                team_picks = get_team_picks(TEAM_ID, CURRENT_GW - 1)
            
            if 'picks' not in team_picks:
                print('Error: Could not retrieve team picks')
                return pd.DataFrame()
            
            my_team_ids = [p['element'] for p in team_picks['picks']]
            my_team = self.players[self.players['id'].isin(my_team_ids)].copy()
            
            print(f'Retrieved current team: {len(my_team)} players')
            return my_team
            
        except Exception as e:
            print(f'Error retrieving current team: {e}')
            return pd.DataFrame()

    def get_team_transfers_info(self):
        '''Get accurate transfer information'''
        try:
            team_info = get_team_info(TEAM_ID)
            
            current_event_free_transfers = team_info.get('current_event_free_transfers', 1)
            bank = team_info.get('last_deadline_bank', 0) / 10.0
            total_transfers = team_info.get('last_deadline_total_transfers', 0)
            
            print(f'Transfer info retrieved:')
            print(f'  Free transfers available: {current_event_free_transfers}')
            print(f'  Bank balance: £{bank:.1f}m')
            print(f'  Total transfers made this season: {total_transfers}')
            
            return current_event_free_transfers, bank, total_transfers
            
        except Exception as e:
            print(f'Error retrieving transfer info: {e}')
            return 1, 0.0, 0

    def analyse_head_to_head_fixtures(self, my_team_df):
        '''Analyse head-to-head matchups between squad players'''
        print('Analysing head-to-head fixtures between your players...')
        
        # Get upcoming fixtures for current gameweek
        current_gw_fixtures = self.fixtures[self.fixtures['event'] == CURRENT_GW]
        
        h2h_conflicts = []
        team_id_to_name = dict(zip(self.teams['id'], self.teams['name']))
        
        # Check for head-to-head matchups
        for _, fixture in current_gw_fixtures.iterrows():
            home_team = fixture['team_h']
            away_team = fixture['team_a']
            
            # Find players from both teams in the squad
            home_players = my_team_df[my_team_df['team'] == home_team]
            away_players = my_team_df[my_team_df['team'] == away_team]
            
            if not home_players.empty and not away_players.empty:
                # Head-to-head conflict identified
                conflict = {
                    'fixture': f'{team_id_to_name.get(home_team, "Unknown")} vs {team_id_to_name.get(away_team, "Unknown")}',
                    'home_team': team_id_to_name.get(home_team, 'Unknown'),
                    'away_team': team_id_to_name.get(away_team, 'Unknown'),
                    'home_players': home_players.copy(),
                    'away_players': away_players.copy(),
                    'total_players': len(home_players) + len(away_players)
                }
                h2h_conflicts.append(conflict)
        
        if not h2h_conflicts:
            print('No head-to-head conflicts found for current gameweek')
            return []
        
        print(f'Found {len(h2h_conflicts)} head-to-head fixture(s) involving your players')
        
        # Calculate priority scores for conflicted players
        for i, conflict in enumerate(h2h_conflicts):
            h2h_conflicts[i] = self._calculate_h2h_priorities(conflict)
        
        return h2h_conflicts

    def _calculate_h2h_priorities(self, conflict):
        '''Calculate priority scores for players in head-to-head conflicts'''
        all_players = pd.concat([conflict['home_players'], conflict['away_players']], ignore_index=True)
        
        # Calculate H2H priority scores
        all_players['h2h_priority_score'] = 0.0
        all_players['is_home_fixture'] = False
        
        # Mark home players
        home_indices = all_players[all_players['id'].isin(conflict['home_players']['id'])].index
        away_indices = all_players[all_players['id'].isin(conflict['away_players']['id'])].index
        
        all_players.loc[home_indices, 'is_home_fixture'] = True
        
        # Calculate priority scores
        for idx, row in all_players.iterrows():
            is_home = row['is_home_fixture']
            priority_score = self._calculate_h2h_priority(row, is_home)
            all_players.loc[idx, 'h2h_priority_score'] = priority_score
        
        # Add opponent information
        all_players['opponent_team'] = ''
        all_players.loc[home_indices, 'opponent_team'] = conflict['away_team']
        all_players.loc[away_indices, 'opponent_team'] = conflict['home_team']
        
        # Sort by priority and select recommended players
        all_players = all_players.sort_values('h2h_priority_score', ascending=False)
        
        conflict['all_players_prioritised'] = all_players
        conflict['recommended_starters'] = all_players.head(2)
        conflict['recommended_bench'] = all_players.tail(len(all_players) - 2) if len(all_players) > 2 else pd.DataFrame()
        
        return conflict

    def _calculate_h2h_priority(self, row, is_home=False):
        '''Calculate H2H priority score for individual player'''
        score = 0
        
        # Base predicted points
        score += row.get('predicted_points', 0) * H2H_WEIGHTS['predicted_points']
        
        # Position-based attacking priority
        if row['position'] in ['Forward', 'Midfielder']:
            score *= H2H_WEIGHTS['attacking_player']
        
        # Home advantage
        if is_home:
            score *= H2H_WEIGHTS['home_advantage']
        
        # Form multiplier
        form_score = (row.get('form_3', 0) + row.get('form_5', 0)) / 2
        if form_score > 5:
            score *= H2H_WEIGHTS['form_multiplier']
        
        # Fixture strength adjustment
        fixture_bonus = 1.0
        if row['position'] == 'Forward':
            fixture_bonus = 1 + (row.get('attacking_favorability', 0.5) - 0.5) * 0.3
        elif row['position'] == 'Midfielder':
            fixture_bonus = 1 + ((row.get('attacking_favorability', 0.5) + row.get('defensive_favorability', 0.5))/2 - 0.5) * 0.2
        elif row['position'] in ['Defender', 'Goalkeeper']:
            fixture_bonus = 1 + (row.get('defensive_favorability', 0.5) - 0.5) * 0.25
        
        score *= fixture_bonus * H2H_WEIGHTS['fixture_strength']
        
        # Consistency bonus
        score += row.get('consistency', 0) * H2H_WEIGHTS['consistency']
        
        # Availability penalty
        score *= row.get('availability_multiplier', 1.0)
        
        return max(0, score)

    def display_h2h_analysis(self, h2h_conflicts):
        '''Display head-to-head analysis results'''
        if not h2h_conflicts:
            return
        
        for i, conflict in enumerate(h2h_conflicts, 1):
            fixture_title = f'H2H CONFLICT {i}: {conflict["fixture"]}'
            
            # Show all players involved with priorities
            all_players_display = conflict['all_players_prioritised'][[
                'full_name', 'position', 'team_name', 'opponent_team', 'is_home_fixture',
                'predicted_points', 'form_5', 'consistency', 'attacking_favorability', 
                'defensive_favorability', 'injury_status', 'h2h_priority_score'
            ]].copy()
            
            # Add home/away indicator
            all_players_display['home_away'] = all_players_display['is_home_fixture'].apply(
                lambda x: 'HOME' if x else 'AWAY'
            )
            
            # Reorder columns for better display
            display_cols = ['full_name', 'position', 'team_name', 'home_away', 'opponent_team',
                           'predicted_points', 'form_5', 'consistency', 'attacking_favorability',
                           'h2h_priority_score', 'injury_status']
            
            clean_table_display(all_players_display[display_cols], fixture_title)
            
            # Show recommendations
            recommended = conflict['recommended_starters']
            if len(recommended) > 1:
                print(f'RECOMMENDATION: Start {recommended.iloc[0]["full_name"]} and {recommended.iloc[1]["full_name"]}')
                print(f'Priority scores: {recommended.iloc[0]["h2h_priority_score"]:.2f} vs {recommended.iloc[1]["h2h_priority_score"]:.2f}')
            elif len(recommended) == 1:
                print(f'RECOMMENDATION: Prioritise {recommended.iloc[0]["full_name"]} (score: {recommended.iloc[0]["h2h_priority_score"]:.2f})')
            
            print(f'Reasoning: Higher priority given to attacking players, home advantage, form, and fixture strength')
            print()

    def suggest_starting_eleven(self, my_team_df, h2h_conflicts=None):
        '''Optimised starting XI selection with head-to-head conflict resolution'''
        print('Optimising starting XI with comprehensive analysis and H2H conflict resolution...')
        
        team_df = my_team_df.copy()
        available_team = team_df[team_df['is_available']].copy()
        
        if len(available_team) < 11:
            print(f'Warning: Only {len(available_team)} available players')
            available_team = team_df.copy()
        
        # Calculate lineup scores
        team_df['lineup_score'] = (
            team_df['predicted_points'] * 0.5 +
            team_df['form_5'] * 0.2 +
            team_df['fixture_favorability'] * 8 * 0.15 +
            team_df['consistency'] * 6 * 0.1 +
            team_df.get('ict_index', 0) * 0.002 * 0.05
        )
        
        # Apply head-to-head adjustments if conflicts exist
        if h2h_conflicts:
            team_df = self._apply_h2h_adjustments(team_df, h2h_conflicts)
        
        available_team = team_df[team_df['is_available']].copy()
        
        position_counts = available_team['position'].value_counts()
        print(f'Available players: {position_counts.to_dict()}')
        
        best_formation = None
        best_score = -1
        best_lineup = None
        
        position_map = {
            'Goalkeeper': 'GK',
            'Defender': 'DEF', 
            'Midfielder': 'MID',
            'Forward': 'FWD'
        }
        
        # Try all valid formations
        for formation in VALID_FORMATIONS:
            lineup = []
            total_score = 0
            formation_valid = True
            
            for pos_abbrev, count in formation.items():
                fpl_position = [k for k, v in position_map.items() if v == pos_abbrev][0]
                pos_players = available_team[available_team['position'] == fpl_position].sort_values(
                    'lineup_score', ascending=False
                )
                
                if len(pos_players) < count:
                    formation_valid = False
                    break
                
                selected = pos_players.head(count)
                lineup.extend(selected.index.tolist())
                total_score += selected['lineup_score'].sum()
            
            if formation_valid and total_score > best_score:
                best_score = total_score
                best_formation = formation
                best_lineup = lineup
        
        # Set player statuses
        team_df['status'] = 'Bench'
        team_df['captain_candidate'] = ''
        
        if best_lineup:
            team_df.loc[best_lineup, 'status'] = 'Starting XI'
            formation_str = f'{best_formation["DEF"]}-{best_formation["MID"]}-{best_formation["FWD"]}'
            print(f'Optimal formation: {formation_str} (Score: {best_score:.1f})')
        
        # Select captain from starting XI
        starting_available = team_df[
            (team_df['status'] == 'Starting XI') & 
            (team_df['is_available'])
        ]
        
        if not starting_available.empty:
            captain_idx = starting_available['lineup_score'].idxmax()
            team_df.loc[captain_idx, 'captain_candidate'] = '(C)'
            
            remaining = starting_available.drop(captain_idx)
            if not remaining.empty:
                vice_captain_idx = remaining['lineup_score'].idxmax()
                team_df.loc[vice_captain_idx, 'captain_candidate'] = '(VC)'
        
        return team_df.sort_values(['status', 'lineup_score'], ascending=[True, False])

    def _apply_h2h_adjustments(self, team_df, h2h_conflicts):
        '''Apply head-to-head priority adjustments to lineup scores'''
        for conflict in h2h_conflicts:
            recommended_starters = conflict['recommended_starters']
            recommended_bench = conflict['recommended_bench']
            
            # Boost recommended starters
            for _, player in recommended_starters.iterrows():
                player_id = player['id']
                boost_factor = 1.3  # Significant boost for recommended starters
                
                mask = team_df['id'] == player_id
                if mask.any():
                    team_df.loc[mask, 'lineup_score'] *= boost_factor
            
            # Reduce score for recommended bench players
            for _, player in recommended_bench.iterrows():
                player_id = player['id']
                reduction_factor = 0.8  # Reduce score for recommended bench
                
                mask = team_df['id'] == player_id
                if mask.any():
                    team_df.loc[mask, 'lineup_score'] *= reduction_factor
        
        return team_df

    def suggest_captaincy(self, my_team_df):
        '''Enhanced captaincy analysis'''
        print('Analysing captaincy with comprehensive metrics...')
        
        captaincy_candidates = my_team_df[
            (my_team_df['is_likely_available']) &
            (my_team_df['predicted_points'] >= 4) &
            (my_team_df['minutes'] >= 60)
        ].copy()
        
        if captaincy_candidates.empty:
            captaincy_candidates = my_team_df[
                (my_team_df['is_available']) &
                (my_team_df['predicted_points'] >= 3)
            ].copy()
        
        if captaincy_candidates.empty:
            print('Warning: No suitable captaincy candidates')
            return pd.DataFrame()
        
        # Calculate captaincy scores
        captaincy_candidates['captaincy_score'] = captaincy_candidates.apply(self._calculate_captaincy_score, axis=1)
        
        captaincy_candidates['risk_level'] = np.where(
            captaincy_candidates['consistency'] > 0.6, 'Low',
            np.where(captaincy_candidates['consistency'] > 0.4, 'Medium', 'High')
        )
        
        return captaincy_candidates.sort_values('captaincy_score', ascending=False)

    def _calculate_captaincy_score(self, row):
        '''Calculate captaincy score for individual player'''
        score = 0
        
        # Base predicted points
        score += row['predicted_points'] * CAPTAINCY_WEIGHTS['predicted_points']
        
        # Fixture scoring with position adjustments
        fixture_bonus = row['fixture_favorability'] * 12
        if row['position'] == 'Forward':
            fixture_bonus += row.get('attacking_favorability', 0.5) * 8
        elif row['position'] == 'Midfielder':
            fixture_bonus += (row.get('attacking_favorability', 0.5) + row.get('defensive_favorability', 0.5)) * 3
        elif row['position'] in ['Defender', 'Goalkeeper']:
            fixture_bonus += row.get('defensive_favorability', 0.5) * 6
        
        score += fixture_bonus * CAPTAINCY_WEIGHTS['fixture_favorability']
        
        # Form assessment
        form_score = (row['form_3'] + row['form_5'] * 0.7) / 1.7
        score += form_score * CAPTAINCY_WEIGHTS['form_trend']
        
        # Consistency bonus
        score += row['consistency'] * 12 * CAPTAINCY_WEIGHTS['consistency']
        
        # Home advantage
        home_bonus = 0.8 if row.get('home_fixtures', 0) > row.get('away_fixtures', 0) else 0
        score += home_bonus * CAPTAINCY_WEIGHTS['home_advantage']
        
        # Additional performance bonuses
        if row.get('ict_index', 0) > 100:
            score += 0.5
        if row.get('expected_goal_involvements', 0) > 0.5:
            score += 0.3
        
        return score * row['availability_multiplier']


def main():
    print('Starting Enhanced FPL Analysis System with Head-to-Head Analysis')
    print('=' * 90)
    
    # Initialise analyser
    analyser = AdvancedFPLAnalyser()
    
    # Run fixture analysis with team strength metrics
    analyser.calculate_fixture_analysis()
    
    # Build comprehensive ML model with realistic bounds
    model, feature_columns = analyser.build_comprehensive_ml_model()
    analyser.generate_comprehensive_predictions(model, feature_columns)
    
    # Calculate comprehensive scores
    analyser.calculate_comprehensive_scores()
    
    # Get current team and transfer info
    free_transfers, bank, total_transfers = analyser.get_team_transfers_info()
    my_team = analyser.get_current_team()
    
    if my_team.empty:
        print('Error: Could not retrieve current team')
        return
    
    # Analyse head-to-head fixtures
    h2h_conflicts = analyser.analyse_head_to_head_fixtures(my_team)
    
    # Display main results
    print(f'\nTEAM ANALYSIS - GW{CURRENT_GW}')
    print(f'Free Transfers: {free_transfers} | Bank: £{bank:.1f}m | Total Transfers: {total_transfers}')
    
    # Display head-to-head analysis
    if h2h_conflicts:
        analyser.display_h2h_analysis(h2h_conflicts)
    
    # Display injury report
    injured_players = my_team[my_team['is_injured']]
    doubtful_players = my_team[my_team['is_doubtful']]
    
    if not injured_players.empty or not doubtful_players.empty:
        print('\n' + '=' * 90)
        print('LIVE INJURY & AVAILABILITY REPORT'.center(90))
        print('=' * 90)
        
        if not injured_players.empty:
            injury_display = injured_players[[
                'full_name', 'position', 'team_name', 'injury_status', 
                'chance_of_playing_this_round', 'news', 'last_gw_points'
            ]]
            clean_table_display(injury_display, 'UNAVAILABLE PLAYERS', max_rows=10)
        
        if not doubtful_players.empty:
            doubtful_display = doubtful_players[[
                'full_name', 'position', 'team_name', 'injury_status',
                'chance_of_playing_this_round', 'predicted_points', 'last_gw_points'
            ]]
            clean_table_display(doubtful_display, 'DOUBTFUL PLAYERS', max_rows=10)
    
    # Current team overview with last GW points
    team_overview = my_team[[
        'full_name', 'position', 'team_name', 'cost', 'last_gw_points', 'predicted_points',
        'form_5', 'consistency', 'fixture_rating', 'injury_status',
        'chance_of_playing_this_round', 'comprehensive_score'
    ]].sort_values('comprehensive_score', ascending=False)
    
    clean_table_display(team_overview, 'CURRENT TEAM OVERVIEW', max_rows=15)
    
    # Captaincy analysis
    captaincy_options = analyser.suggest_captaincy(my_team)
    if not captaincy_options.empty:
        captaincy_display = captaincy_options[[
            'full_name', 'position', 'last_gw_points', 'predicted_points', 'form_3', 'form_5',
            'fixture_rating', 'attacking_favorability', 'consistency', 'injury_status', 'captaincy_score', 'risk_level'
        ]].head(5)
        clean_table_display(captaincy_display, 'CAPTAINCY RECOMMENDATIONS', max_rows=5)
    
    # Starting XI with head-to-head consideration
    lineup_suggestions = analyser.suggest_starting_eleven(my_team, h2h_conflicts)
    starting_xi = lineup_suggestions[lineup_suggestions['status'] == 'Starting XI']
    bench = lineup_suggestions[lineup_suggestions['status'] == 'Bench']
    
    if not starting_xi.empty:
        starting_display = starting_xi[[
            'full_name', 'position', 'last_gw_points', 'predicted_points', 'lineup_score',
            'form_5', 'fixture_rating', 'injury_status', 'captain_candidate'
        ]]
        clean_table_display(starting_display, 'RECOMMENDED STARTING XI (H2H OPTIMISED)', max_rows=11)
    
    if not bench.empty:
        bench_display = bench[[
            'full_name', 'position', 'last_gw_points', 'predicted_points', 'lineup_score', 
            'form_5', 'injury_status'
        ]]
        clean_table_display(bench_display, 'RECOMMENDED BENCH', max_rows=4)
    
    # Transfer analysis
    candidates = analyser.players[
        (~analyser.players['id'].isin(my_team['id'])) &
        (analyser.players['is_likely_available']) &
        (analyser.players['minutes'] > 45)
    ].copy()
    
    locked_ids = my_team[my_team['full_name'].isin(LOCKED_PLAYERS)]['id'].tolist()
    
    priority_transfers = []
    regular_transfers = []
    
    # Generate transfer recommendations
    for _, player in my_team.iterrows():
        if player['id'] in locked_ids:
            continue
        
        max_budget = bank + player['cost']
        same_pos_candidates = candidates[
            (candidates['position'] == player['position']) & 
            (candidates['cost'] <= max_budget)
        ]
        
        if same_pos_candidates.empty:
            continue
        
        if player['is_injured'] or player['is_doubtful']:
            min_improvement = 0
            transfer_type = 'priority'
        else:
            min_improvement = TRANSFER_STRATEGY['min_score_improvement']
            transfer_type = 'regular'
        
        viable_candidates = same_pos_candidates[
            same_pos_candidates['comprehensive_score'] > player['comprehensive_score'] + min_improvement
        ]
        
        if viable_candidates.empty:
            continue
        
        best_options = viable_candidates.sort_values(
            ['comprehensive_score', 'value_score'], ascending=False
        ).head(3)
        
        for _, candidate in best_options.iterrows():
            score_gain = candidate['comprehensive_score'] - player['comprehensive_score']
            transfer_info = {
                'out_player': player['full_name'],
                'out_position': player['position'],
                'out_injury_status': player['injury_status'],
                'out_score': player['comprehensive_score'],
                'out_last_gw': player['last_gw_points'],
                'in_player': candidate['full_name'],
                'in_cost': candidate['cost'],
                'in_score': candidate['comprehensive_score'],
                'in_last_gw': candidate['last_gw_points'],
                'score_gain': score_gain,
                'ownership': candidate['ownership'],
                'form_trend': candidate['trend'],
                'fixture_rating': candidate['fixture_rating'],
                'attacking_favorability': candidate.get('attacking_favorability', 0.5),
                'transfer_urgency': 'HIGH' if player['is_injured'] else 'MEDIUM' if player['is_doubtful'] else 'LOW'
            }
            
            if transfer_type == 'priority':
                priority_transfers.append(transfer_info)
            else:
                regular_transfers.append(transfer_info)
    
    # Display priority transfers
    if priority_transfers:
        priority_df = pd.DataFrame(priority_transfers)
        priority_display = priority_df[[
            'out_player', 'out_injury_status', 'out_last_gw', 'in_player', 'in_cost', 
            'in_last_gw', 'score_gain', 'transfer_urgency', 'fixture_rating', 'attacking_favorability'
        ]]
        clean_table_display(priority_display, 'PRIORITY TRANSFERS (Injured/Doubtful Players)', max_rows=10)
    
    # Display regular transfers
    if regular_transfers:
        regular_df = pd.DataFrame(sorted(regular_transfers, key=lambda x: x['score_gain'], reverse=True)[:10])
        regular_display = regular_df[[
            'out_player', 'out_last_gw', 'in_player', 'in_cost', 'in_last_gw',
            'score_gain', 'ownership', 'form_trend', 'fixture_rating', 'attacking_favorability'
        ]]
        clean_table_display(regular_display, 'STRATEGIC TRANSFER RECOMMENDATIONS', max_rows=10)
    
    # Display market categories
    _display_market_categories(candidates)
    
    # Strategic insights
    _display_strategic_insights(h2h_conflicts, my_team, priority_transfers, free_transfers)
    
    # Display chip strategy
    display_chip_strategy()
    
    print('\n' + '=' * 90)
    print('ANALYSIS COMPLETE - Enhanced with Head-to-Head fixture optimisation!')
    print('=' * 90)


def _display_market_categories(candidates):
    '''Display different market categories for transfers'''
    
    # Differential picks
    differential_gems = candidates[
        (candidates['ownership'] < TRANSFER_STRATEGY['ownership_differential_threshold']) &
        (candidates['predicted_points'] > 4) &
        (candidates['form_5'] > 3) &
        (candidates['comprehensive_score'] > 30)
    ].sort_values('differential_score', ascending=False).head(20)
    
    if not differential_gems.empty:
        differential_display = differential_gems[[
            'full_name', 'position', 'team_name', 'cost', 'last_gw_points', 'predicted_points',
            'form_5', 'ownership', 'fixture_rating', 'attacking_favorability', 'comprehensive_score', 'injury_status'
        ]]
        clean_table_display(differential_display, 'DIFFERENTIAL PICKS (Low Ownership Gems)', max_rows=20)
    
    # Form players
    form_players = candidates[
        (candidates['trend'] == 'rising') &
        (candidates['form_5'] > candidates['form_10']) &
        (candidates['predicted_points'] > 4) &
        (candidates['momentum'] > 0.1)
    ].sort_values(['momentum', 'form_5'], ascending=False).head(15)
    
    if not form_players.empty:
        form_display = form_players[[
            'full_name', 'position', 'team_name', 'cost', 'last_gw_points', 'predicted_points',
            'form_3', 'form_5', 'momentum', 'fixture_rating', 'attacking_favorability', 'injury_status'
        ]]
        clean_table_display(form_display, 'HOT FORM PLAYERS', max_rows=15)
    
    # Value picks
    value_picks = candidates[
        (candidates['value_score'] > candidates['value_score'].quantile(0.8)) &
        (candidates['predicted_points'] > 3.5) &
        (candidates['cost'] <= 7.0)
    ].sort_values('value_score', ascending=False).head(15)
    
    if not value_picks.empty:
        value_display = value_picks[[
            'full_name', 'position', 'team_name', 'cost', 'last_gw_points', 'predicted_points',
            'value_score', 'form_5', 'fixture_rating', 'attacking_favorability', 'ownership'
        ]]
        clean_table_display(value_display, 'BEST VALUE PICKS', max_rows=15)
    
    # Premium options
    premium_picks = candidates[
        (candidates['cost'] >= 8.0) &
        (candidates['predicted_points'] > 6) &
        (candidates['comprehensive_score'] > 50)
    ].sort_values('comprehensive_score', ascending=False).head(15)
    
    if not premium_picks.empty:
        premium_display = premium_picks[[
            'full_name', 'position', 'team_name', 'cost', 'last_gw_points', 'predicted_points',
            'form_5', 'fixture_rating', 'attacking_favorability', 'comprehensive_score', 'ownership'
        ]]
        clean_table_display(premium_display, 'PREMIUM OPTIONS (8.0M+)', max_rows=15)


def _display_strategic_insights(h2h_conflicts, my_team, priority_transfers, free_transfers):
    '''Display comprehensive strategic insights'''
    print('\n' + '=' * 90)
    print('COMPREHENSIVE STRATEGIC INSIGHTS WITH H2H ANALYSIS'.center(90))
    print('=' * 90)
    
    # Head-to-head insights
    if h2h_conflicts:
        print(f'\nHEAD-TO-HEAD FIXTURE ANALYSIS:')
        print(f'   Found {len(h2h_conflicts)} fixture(s) with squad conflicts')
        for i, conflict in enumerate(h2h_conflicts, 1):
            print(f'   Conflict {i}: {conflict["fixture"]} ({conflict["total_players"]} players involved)')
        print('   Starting XI optimised to prioritise best performers in H2H situations')
        print('   Recommendation: Monitor team news closely for conflicted players')
    else:
        print(f'\nHEAD-TO-HEAD FIXTURE ANALYSIS:')
        print('   No head-to-head conflicts found this gameweek')
        print('   Standard lineup optimisation applied')
    
    # Prediction quality check
    total_predicted = my_team[my_team['is_likely_available']]['predicted_points'].sum()
    last_gw_total = my_team['last_gw_points'].sum()
    
    print(f'\nPREDICTION QUALITY CHECK:')
    print(f'   Expected points this GW: {total_predicted:.1f}')
    print(f'   Last GW total: {last_gw_total}')
    print(f'   Prediction seems: {"Realistic" if total_predicted < 80 else "High" if total_predicted < 100 else "TOO HIGH - Check model"}')
    
    # Injury impact analysis
    total_injured = len(my_team[my_team['is_injured']])
    total_doubtful = len(my_team[my_team['is_doubtful']])
    
    print(f'\nINJURY IMPACT ANALYSIS:')
    print(f'   Injured/Suspended players: {total_injured}')
    print(f'   Doubtful players: {total_doubtful}')
    print(f'   Fully available players: {len(my_team[my_team["is_likely_available"]])}')
    
    if total_injured > 0:
        print(f'   URGENT: {total_injured} player(s) unavailable - immediate transfers needed')
    if total_doubtful > 2:
        print(f'   CAUTION: {total_doubtful} players doubtful - monitor team news closely')
    
    # Transfer priority guidance
    print(f'\nTRANSFER PRIORITY GUIDANCE:')
    if priority_transfers:
        unique_priority_players = len(set([t['out_player'] for t in priority_transfers]))
        print(f'   URGENT: Replace {unique_priority_players} injured/doubtful players')
        print(f'   Recommended: Use all {free_transfers} free transfer(s) this week')
        if free_transfers < unique_priority_players:
            points_hit = (unique_priority_players - free_transfers) * 4
            print(f'   Consider taking {points_hit}-point hit for essential transfers')
    else:
        print('   No urgent transfers needed due to injuries')
        print(f'   Optional improvements available - save transfers or upgrade')
    
    # Fixture analysis summary
    good_attacking_fixtures = len(my_team[my_team.get('attacking_favorability', 0.5) > 0.6])
    good_defensive_fixtures = len(my_team[my_team.get('defensive_favorability', 0.5) > 0.6])
    
    print(f'\nFIXTURE ANALYSIS SUMMARY:')
    print(f'   Players with good attacking fixtures: {good_attacking_fixtures}')
    print(f'   Players with good defensive fixtures: {good_defensive_fixtures}')
    print(f'   Players with overall good fixtures: {len(my_team[my_team["fixture_favorability"] > 0.6])}')
    
    if good_attacking_fixtures >= 6:
        print('   Strong attacking fixtures - good captain options available')
    if good_defensive_fixtures >= 4:
        print('   Good clean sheet potential this gameweek')


if __name__ == '__main__':
    main()