# Fantasy-Premier-League-Machine-Learning-Algorithm
<hr>
An algorithm made with Python using machine learning to predict the best picks for Fantasy Premier League

## SCROLL TO THE BOTTOM TO SEE RESULTS (MACHINE LEARNING ACCURACY, TEAM OVERVIEWS, BEST PICKS, DIFFERENTIAL PICKS, CAPTAINCY RECOMMENDATIONS, STARTING XI RECOMMENDATIONS)

# Advanced Fantasy Premier League Analysis System
<hr>
A comprehensive machine learning-driven analysis tool for Fantasy Premier League that combines real-time API data, advanced statistical modelling, and strategic optimisation to provide actionable insights for team management and transfer decisions.

## Project Overview
<hr>
This system represents a sophisticated approach to Fantasy Premier League analysis, integrating multiple data sources and analytical techniques to solve complex decision-making problems in sports analytics. The project demonstrates proficiency in API integration, machine learning implementation, data processing, and strategic algorithm development.

### Key Features

- **Real-time API Integration**: Live data retrieval from the official FPL API including player statistics, fixtures, injuries, and team information
- **Machine Learning Predictions**: Advanced ensemble models (Random Forest and Gradient Boosting) trained on comprehensive historical data
- **Head-to-Head Analysis**: Strategic evaluation of fixture conflicts between squad players
- **Transfer Optimisation**: Multi-criteria decision analysis for player transfers considering form, fixtures, value, and availability
- **Dynamic Team Selection**: Formation-aware starting XI optimisation with tactical considerations
- **Injury Management**: Real-time availability tracking with risk assessment
- **Captaincy Analysis**: Multi-factor scoring system for captain selection
- **Market Intelligence**: Player categorisation and differential identification

## Technical Architecture

### Data Processing Pipeline
<hr>
The system follows a robust data processing pipeline that handles multiple data sources and transformation stages:

1. **API Data Ingestion**: Structured retrieval from FPL endpoints with error handling and rate limiting
2. **Data Validation**: Comprehensive cleaning and validation of incoming data
3. **Feature Engineering**: Creation of derived metrics and statistical indicators
4. **Model Training**: Dynamic model selection based on data availability and quality
5. **Prediction Generation**: Real-time scoring with confidence intervals and bounds checking

### Machine Learning Implementation
<hr>
The predictive engine employs several sophisticated techniques:

**Model Selection Strategy**: The system implements dynamic model selection based on available training data volume. For smaller datasets, it uses Random Forest for stability, whilst larger datasets benefit from Gradient Boosting with optimised hyperparameters.

**Feature Engineering**: Over 40 features are engineered from raw FPL data, including:
- Expected goal involvement metrics (xG, xA, xGI)
- Form trends with weighted recent performance
- Fixture difficulty and strength analysis
- Positional performance indicators
- Availability and injury risk factors

**Training Data Generation**: The system processes historical player performance data, handling missing values and outliers through robust statistical methods. Training samples are generated with realistic bounds to prevent overfitting and ensure model reliability.

**Prediction Validation**: Multiple validation layers ensure predictions remain within realistic bounds, with regression-to-mean adjustments and fixture-based modifications.

### Strategic Algorithms

**Head-to-Head Analysis**: A unique algorithm that identifies when multiple squad players face each other in fixtures, calculating priority scores based on positional advantages, home/away factors, and current form to optimise lineup decisions.

**Transfer Optimisation**: Multi-objective optimisation considering:
- Performance improvement potential
- Budget constraints and value analysis
- Ownership differentials for competitive advantage
- Injury urgency and availability windows

**Formation Selection**: Dynamic evaluation of all valid FPL formations (3-4-3, 3-5-2, 4-3-3, 4-4-2, 4-5-1, 5-3-2, 5-4-1) with position-specific scoring to identify optimal tactical setups.

## Code Architecture and Design Patterns

### Object-Oriented Design

The core `AdvancedFPLAnalyser` class demonstrates solid object-oriented principles:
- **Encapsulation**: Private methods for internal calculations and data processing
- **Modularity**: Separate methods for distinct analytical functions
- **Extensibility**: Clean interfaces allowing for future enhancements

### Error Handling and Robustness
<hr>
Comprehensive error handling throughout the system:
- API request failures with graceful degradation
- Data validation with fallback mechanisms
- Model training failures with alternative approaches
- Missing data imputation using statistical methods

### Performance Optimisation

Several optimisation strategies are implemented:
- **Sampling Strategy**: Intelligent sampling of large datasets to balance accuracy with processing time
- **Vectorised Operations**: Use of NumPy and Pandas for efficient mathematical operations
- **Memory Management**: Careful handling of large datasets with selective loading
- **Caching**: Implied caching strategies for repeated calculations

## Data Science Methodology

### Statistical Analysis

The system employs rigorous statistical methods:
- **Normalisation**: Z-score normalisation for cross-metric comparison
- **Trend Analysis**: Time-series analysis for form and momentum calculations
- **Risk Assessment**: Variance-based consistency metrics
- **Confidence Intervals**: Prediction bounds based on historical accuracy

### Feature Selection and Importance
<hr>
Advanced feature selection techniques identify the most predictive variables:
- Analysis of feature importance from ensemble models
- Cross-validation to prevent overfitting
- Domain knowledge integration for football-specific insights

### Model Evaluation
<hr>
Comprehensive model evaluation using multiple metrics:
- R-squared for explained variance
- Mean Absolute Error for prediction accuracy
- Custom football-specific validation metrics

## Strategic Intelligence Features

### Market Analysis

The system provides sophisticated market intelligence:
- **Differential Analysis**: Identification of low-ownership high-potential players
- **Value Identification**: Price-to-performance ratio calculations
- **Form Trending**: Detection of players entering or leaving good form periods
- **Premium Analysis**: Evaluation of expensive players' value propositions

### Fixture Analysis

Advanced fixture difficulty assessment:
- **Team Strength Modelling**: Dynamic calculation of attacking and defensive strength
- **Positional Adjustments**: Different fixture evaluations for different positions
- **Home/Away Factors**: Statistical modelling of venue advantages
- **Multi-Gameweek Planning**: Forward-looking fixture analysis

### Injury Intelligence

Real-time injury management system:
- **Availability Tracking**: Live monitoring of player fitness status
- **Risk Categorisation**: Classification of injury severity and return likelihood
- **Replacement Prioritisation**: Urgent vs. strategic transfer identification

## Technical Skills Demonstrated

### Programming Proficiency
- **Python**: Advanced use of Python for data science applications
- **API Integration**: RESTful API consumption with error handling
- **Data Structures**: Efficient use of dictionaries, lists, and pandas DataFrames
- **Object-Oriented Programming**: Clean class design with appropriate method organisation

### Data Science and Machine Learning
- **Feature Engineering**: Creation of meaningful variables from raw data
- **Model Selection**: Appropriate algorithm choice based on data characteristics
- **Cross-Validation**: Proper model evaluation techniques
- **Hyperparameter Tuning**: Optimisation of model parameters

### Problem-Solving Approach
- **Domain Analysis**: Deep understanding of Fantasy Premier League mechanics
- **Multi-Objective Optimisation**: Balancing competing priorities in decision-making
- **Risk Management**: Handling uncertainty in predictions and recommendations
- **Strategic Thinking**: Long-term planning with short-term tactical adjustments

### Software Engineering Practices
- **Code Documentation**: Comprehensive commenting and function documentation
- **Error Handling**: Robust exception handling and graceful degradation
- **Modularity**: Well-structured code with clear separation of concerns
- **Configuration Management**: Centralised configuration with easy parameter adjustment

## Usage and Output
<hr>
The system provides comprehensive analysis output including:

**Team Analysis**: Current squad evaluation with performance metrics and availability status

**Injury Reports**: Real-time injury and suspension tracking with impact assessment

**Transfer Recommendations**: Priority-ranked transfer suggestions with supporting analysis

**Starting XI Optimisation**: Formation-specific team selection with tactical reasoning

**Captaincy Analysis**: Multi-factor captain recommendations with risk assessment

**Market Intelligence**: Identification of differential picks, form players, and value opportunities

**Strategic Insights**: High-level strategic recommendations with supporting data

## Future Development Opportunities
<hr>
The modular architecture allows for several enhancement opportunities:
- Integration with additional data sources (player tracking data, weather conditions)
- Advanced visualisation capabilities
- Web application development for user interface
- Database integration for historical data storage
- Real-time alert systems for injury updates and price changes

## Installation and Configuration
<hr>
The system requires Python 3.7+ with the following key dependencies:
- `requests` for API communication
- `pandas` and `numpy` for data manipulation
- `scikit-learn` for machine learning capabilities
- `warnings` for clean output management

Configuration is managed through constants at the top of the script, allowing easy customisation of:
- Team ID for analysis
- Current gameweek
- Prediction parameters
- Transfer strategy settings
- Analysis depth and sampling sizes

## Conclusion
<hr>
This project demonstrates a comprehensive approach to sports analytics, combining machine learning, statistical analysis, and domain expertise to create actionable insights. The system showcases technical proficiency across multiple areas whilst solving real-world problems in competitive fantasy sports management.

The code represents production-quality software development practices with robust error handling, comprehensive documentation, and modular architecture suitable for maintenance and enhancement in professional environments.

## Results
I have been using the algorithm for the past 3 gameweeks, and will continue to allow it to influence my decisions for the 2025/26 Premier League season. So far, I have performed average in Gameweek 1 and 3, but scored 73 points in gameweek 2 (43% gain from global average). Unfortunately, I have dropped some places in my leagues after the average performance in Gameweek 3.



<img width="552" height="569" alt="Screenshot 2025-09-09 at 15 00 15" src="https://github.com/user-attachments/assets/081760ec-195d-4943-b704-cc9c07ebc355" />

<img width="1165" height="397" alt="Screenshot 2025-09-09 at 15 00 37" src="https://github.com/user-attachments/assets/0b50b556-9b66-4ef9-b53f-656d32aa471b" />

<img width="1302" height="838" alt="Screenshot 2025-09-09 at 15 00 57" src="https://github.com/user-attachments/assets/a08a4faf-e4e0-4a67-b782-868a4607d5b7" />

<img width="1205" height="761" alt="Screenshot 2025-09-09 at 15 01 15" src="https://github.com/user-attachments/assets/df8b6bf6-d037-4034-9cce-6ea4307c6ead" />

<img width="1130" height="495" alt="Screenshot 2025-09-09 at 15 01 33" src="https://github.com/user-attachments/assets/189ba926-6192-4d21-aedd-c880f3d26b3c" />

<img width="673" height="451" alt="Screenshot 2025-09-09 at 15 01 52" src="https://github.com/user-attachments/assets/ce6787e9-c620-436c-b185-bce25e869e19" />

<img width="697" height="726" alt="Screenshot 2025-09-09 at 15 02 07" src="https://github.com/user-attachments/assets/1137c6db-daaf-4c80-844f-2a75755df28d" />
