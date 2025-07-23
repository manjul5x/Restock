# Demand Visualization Web Application

A modern web interface for exploring demand data with interactive filtering and time aggregation capabilities.

## Features

- **Interactive Filtering**: Filter by locations, categories, products, and date ranges
- **Time Aggregation**: View data at daily, weekly, or monthly levels
- **Multiple Plot Types**: Demand trends, category comparisons, and location comparisons
- **Real-time Plot Generation**: Generate plots on-demand with your selected filters
- **Data Summary Dashboard**: Overview of key metrics and data statistics

## Setup

1. **Install Dependencies**:
   ```bash
   pip install flask matplotlib seaborn pandas numpy pydantic
   ```

2. **Run the Application**:
   ```bash
   # From the project root directory
   python webapp/run.py
   
   # Or from the webapp directory
   cd webapp
   python run.py
   ```

3. **Access the Application**:
   Open your browser and go to: http://localhost:8080

## Usage

1. **Select Plot Type**: Choose between Demand Trend, Category Comparison, or Location Comparison
2. **Set Time Aggregation**: Choose daily, weekly, or monthly aggregation
3. **Filter Data**: Select specific locations, categories, products, and date ranges
4. **Generate Plot**: Click "Generate Plot" to create your visualization
5. **Explore**: The plot will display with your selected filters and aggregation

## File Structure

```
webapp/
├── __init__.py          # Package initialization
├── app.py              # Flask application
├── run.py              # Run script
├── README.md           # This file
├── templates/          # HTML templates
│   └── index.html      # Main dashboard template
└── static/             # Static files (CSS, JS, images)
```

## API Endpoints

- `GET /`: Main dashboard page
- `POST /generate_plot`: Generate plot based on form data
- `GET /get_data_summary`: Get data summary statistics

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy
- **Data Validation**: Pydantic 