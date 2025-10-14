# Inventory Metrics Guide

## Overview
This guide explains how inventory metrics are calculated, where they are defined, and how to add or modify metrics in the system.

## Metric Architecture

The metrics system is built on three main components:

1. **Metric Definitions** (`forecaster/utils/metric_definitions.py`)
   - Contains standardized definitions for all metrics
   - Includes descriptions, calculations, targets, and typical ranges
   - Provides helper methods for accessing metric information

2. **Metric Calculator** (`forecaster/utils/inventory_metrics_calculator.py`)
   - Handles actual calculation of metrics using vectorized operations
   - Contains validation logic for input data
   - Provides both individual and batch calculation methods

3. **Web Application** (`webapp/app.py`)
   - Displays metrics in the UI
   - Handles aggregation of metrics across products/locations
   - Manages metric visualization and comparison

## Current Metrics

Each metric in `metric_definitions.py` includes:
- Description
- Calculation formula
- Target value or range
- Better when (higher/lower/optimal)
- Unit of measurement
- Typical range

Example:
```python
'inventory_turnover_ratio': {
    'description': 'Number of times inventory is sold and replaced over a period',
    'calculation': 'Cost of Goods Sold (actual demand * unit cost) / Cost of average inventory',
    'target': 'varies by industry',
    'better_when': 'higher',
    'unit': 'ratio',
    'typical_range': (4, 12)
}
```

## Steps to Add a New Metric

1. **Define the Metric**
   - Add metric definition to `metric_definitions.py`
   - Include all required fields (description, calculation, target, etc.)
   - Consider typical ranges and when the metric is "better"

2. **Implement Calculation**
   - Add calculation logic in `inventory_metrics_calculator.py`
   - Use vectorized operations (numpy/pandas) for performance
   - Add to `calculate_all_metrics_vectorized` method
   - Include the metric in the return dictionary

3. **Add to Web Application**
   - Update `get_comparison_data` in `webapp/app.py` to include the metric
   - Add to overall metrics calculation if needed
   - Update templates to display the new metric

4. **Add Tests**
   - Add test cases in `test_inventory_metrics.py`
   - Include edge cases and validation tests
   - Test aggregation if applicable

## Steps to Modify an Existing Metric

1. **Update Definition**
   - Modify the metric definition in `metric_definitions.py`
   - Update description and calculation formula if changed
   - Consider impacts on typical ranges and targets

2. **Modify Calculation**
   - Update calculation logic in `inventory_metrics_calculator.py`
   - Ensure vectorized operations are maintained
   - Update any dependent calculations

3. **Update Web Application**
   - Modify aggregation logic if needed in `webapp/app.py`
   - Update any display formatting or visualization
   - Consider impacts on overall metrics

4. **Update Tests**
   - Modify existing test cases
   - Add new test cases for changed behavior
   - Verify edge cases still work

## Best Practices

1. **Vectorization**
   - Always use vectorized operations (numpy/pandas)
   - Avoid loops for performance
   - Use broadcasting where possible

2. **Validation**
   - Add input validation for new metrics
   - Check for edge cases (zeros, nulls)
   - Validate output ranges

3. **Documentation**
   - Keep metric definitions clear and complete
   - Document calculation methods
   - Include examples where helpful

4. **Testing**
   - Test individual calculations
   - Test aggregations
   - Test edge cases
   - Test with real data samples

## Adding Metrics to the UI

To display a new metric in the web interface, you need to modify two sections in the `inventory_comparison.html` template:

1. **Overall Performance Summary**
   ```html
   <!-- Add to the metrics section -->
   <div class="metric-line" id="your-metric-line">
       <div class="metric-comparison">
           <span class="metric-label">Your Metric:</span>
           <span class="actual-value" id="actual-your-metric">-</span>
           <span class="performance-indicator" id="your-metric-indicator"></span>
       </div>
   </div>
   ```

2. **Detailed Performance Comparison Table**
   - Add column header in the table header section
   ```html
   <th colspan="1" class="text-center">Your Metric</th>
   ```
   - Add column in the table header row
   ```html
   <th class="text-success">Actual</th>
   ```
   - Update the JavaScript table row generation
   ```javascript
   <td class="text-success">${result.your_metric_name}</td>
   ```

3. **Update JavaScript**
   ```javascript
   document.getElementById('actual-your-metric').textContent = data.overall_metrics.avg_your_metric;
   applyHighlighting('your-metric-line', data.overall_metrics.avg_your_metric, 
                     null, true/false, 'your-metric-indicator');
   ```

Remember to:
- Use consistent styling with existing metrics
- Add appropriate tooltips if needed
- Consider the metric's "better when" direction for highlighting
- Place the metric in a logical group with related metrics

## Example: Adding a New Metric

Here's an example of adding a new "Days of Supply" metric:

1. **Add Definition**
```python
'days_of_supply': {
    'description': 'Number of days current inventory will last based on forecast',
    'calculation': 'current_inventory / average_daily_forecast',
    'target': '30-60 days',
    'better_when': 'optimal',
    'unit': 'days',
    'typical_range': (15, 90)
}
```

2. **Add Calculation**
```python
def calculate_all_metrics_vectorized(self, group: pd.DataFrame, inventory_cost: float) -> Dict[str, Any]:
    # ... existing code ...
    
    # Calculate days of supply
    avg_daily_forecast = np.mean(arrays['FRSP'])
    current_inventory = arrays['inventory_on_hand'][-1]
    days_of_supply = current_inventory / avg_daily_forecast if avg_daily_forecast > 0 else 0
    
    return {
        # ... existing metrics ...
        "days_of_supply": round(days_of_supply, 2)
    }
```

3. **Add to Web Application**
```python
# In get_comparison_data
overall_metrics = {
    # ... existing metrics ...
    "avg_days_of_supply": round(
        sum(r["days_of_supply"] for r in comparison_results)
        / len(comparison_results),
        2,
    )
}
```

## Common Issues and Solutions

1. **Performance Issues**
   - Use vectorized operations
   - Minimize DataFrame copies
   - Profile calculations if needed

2. **Data Quality**
   - Add validation checks
   - Handle missing data appropriately
   - Document assumptions

3. **Edge Cases**
   - Handle division by zero
   - Consider negative values
   - Test boundary conditions

## Getting Help

For questions or issues:
1. Check existing metric implementations for examples
2. Review test cases for similar metrics
3. Consult the codebase documentation
4. Reach out to the development team
