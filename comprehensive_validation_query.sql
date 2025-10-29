-- =================================================================================
-- COMPREHENSIVE VALIDATION QUERY
-- =================================================================================
-- This query validates the complete pipeline for 5 random products:
-- 1. Raw data from STAGE schema
-- 2. Processed data from TRANSFORMATION schema  
-- 3. Forecast data from TRANSFORMATION schema
-- =================================================================================

WITH random_products AS (
    -- Select 5 random products for validation
    SELECT DISTINCT PRODUCT_ID
    FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN
    ORDER BY RANDOM()
    LIMIT 5
),

-- Get raw data for these products
raw_data AS (
    SELECT 
        rp.PRODUCT_ID,
        COUNT(*) as raw_records,
        MIN(DATE) as raw_start_date,
        MAX(DATE) as raw_end_date,
        AVG(DEMAND) as avg_demand,
        AVG(UNIT_PRICE) as avg_unit_price,
        AVG(STOCK_LEVEL) as avg_stock_level,
        SUM(DEMAND) as total_demand
    FROM random_products rp
    JOIN RESTOCK_DB.STAGE.OUTFLOW_BHASIN o ON rp.PRODUCT_ID = o.PRODUCT_ID
    GROUP BY rp.PRODUCT_ID
),

-- Get processed data for these products
processed_data AS (
    SELECT 
        rp.PRODUCT_ID,
        COUNT(*) as processed_records,
        MIN(DATE) as processed_start_date,
        MAX(DATE) as processed_end_date,
        AVG(DEMAND) as avg_demand_processed,
        AVG(OUTFLOW) as avg_outflow,
        AVG(RP_LAG) as avg_rp_lag,
        AVG(HALF_RP_LAG) as avg_half_rp_lag,
        AVG(SEASON) as avg_season,
        AVG(SEASON2) as avg_season2,
        AVG(WEEK_1) as avg_week_1,
        AVG(WEEK_2) as avg_week_2,
        AVG(WEEK_3) as avg_week_3,
        AVG(WEEK_4) as avg_week_4,
        AVG(RECENCY) as avg_recency
    FROM random_products rp
    JOIN RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS p ON rp.PRODUCT_ID = p.PRODUCT_ID
    GROUP BY rp.PRODUCT_ID
),

-- Get forecast data for these products
forecast_data AS (
    SELECT 
        rp.PRODUCT_ID,
        PREDICTED_OUTFLOW_TOTAL,
        PREDICTED_OUTFLOW_DAILY,
        RISK_PERIOD_DAYS,
        TRAINING_DATA_POINTS,
        FORECAST_METHOD,
        DEMAND_FREQUENCY,
        FORECASTED_ON,
        FORECAST_PERIOD_START,
        FORECAST_PERIOD_END
    FROM random_products rp
    JOIN RESTOCK_DB.TRANSFORMATION.FUTURE_PREDICTIONS_RESULTS f ON rp.PRODUCT_ID = f.PRODUCT_ID
)

-- Final validation results
SELECT 
    r.PRODUCT_ID,
    
    -- Raw data validation
    r.raw_records,
    r.raw_start_date,
    r.raw_end_date,
    r.avg_demand,
    r.avg_unit_price,
    r.avg_stock_level,
    r.total_demand,
    
    -- Processed data validation
    p.processed_records,
    p.processed_start_date,
    p.processed_end_date,
    p.avg_demand_processed,
    p.avg_outflow,
    p.avg_rp_lag,
    p.avg_half_rp_lag,
    p.avg_season,
    p.avg_season2,
    p.avg_week_1,
    p.avg_week_2,
    p.avg_week_3,
    p.avg_week_4,
    p.avg_recency,
    
    -- Forecast data validation
    f.PREDICTED_OUTFLOW_TOTAL,
    f.PREDICTED_OUTFLOW_DAILY,
    f.RISK_PERIOD_DAYS,
    f.TRAINING_DATA_POINTS,
    f.FORECAST_METHOD,
    f.DEMAND_FREQUENCY,
    f.FORECASTED_ON,
    f.FORECAST_PERIOD_START,
    f.FORECAST_PERIOD_END,
    
    -- Validation checks
    CASE 
        WHEN r.raw_records = p.processed_records THEN '✅ MATCH'
        ELSE '❌ MISMATCH'
    END as record_count_validation,
    
    CASE 
        WHEN ABS(r.avg_demand - p.avg_demand_processed) < 0.001 THEN '✅ MATCH'
        ELSE '❌ MISMATCH'
    END as demand_validation,
    
    CASE 
        WHEN ABS(f.PREDICTED_OUTFLOW_TOTAL / f.RISK_PERIOD_DAYS - f.PREDICTED_OUTFLOW_DAILY) < 0.001 THEN '✅ MATCH'
        ELSE '❌ MISMATCH'
    END as forecast_calculation_validation,
    
    CASE 
        WHEN f.TRAINING_DATA_POINTS = r.raw_records THEN '✅ MATCH'
        ELSE '❌ MISMATCH'
    END as training_data_validation,
    
    CASE 
        WHEN p.avg_rp_lag > 0 AND p.avg_half_rp_lag > 0 THEN '✅ VALID'
        ELSE '❌ INVALID'
    END as regressor_validation

FROM raw_data r
JOIN processed_data p ON r.PRODUCT_ID = p.PRODUCT_ID
JOIN forecast_data f ON r.PRODUCT_ID = f.PRODUCT_ID
ORDER BY r.PRODUCT_ID;

-- =================================================================================
-- BONUS QUERY: SAMPLE DATA FOR EACH PRODUCT
-- =================================================================================
-- This shows sample raw and processed data for the same products

WITH random_products AS (
    SELECT DISTINCT PRODUCT_ID
    FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN
    ORDER BY RANDOM()
    LIMIT 5
)

SELECT 
    'RAW_DATA' as data_type,
    rp.PRODUCT_ID,
    o.DATE,
    o.DEMAND,
    o.UNIT_PRICE,
    o.STOCK_LEVEL,
    o.INCOMING_INVENTORY,
    o.PRODUCT_CATEGORY
FROM random_products rp
JOIN RESTOCK_DB.STAGE.OUTFLOW_BHASIN o ON rp.PRODUCT_ID = o.PRODUCT_ID
WHERE o.DATE = (SELECT MAX(DATE) FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN WHERE PRODUCT_ID = rp.PRODUCT_ID)

UNION ALL

SELECT 
    'PROCESSED_DATA' as data_type,
    rp.PRODUCT_ID,
    p.DATE,
    p.DEMAND,
    p.OUTFLOW,
    p.RP_LAG,
    p.HALF_RP_LAG,
    p.SEASON
FROM random_products rp
JOIN RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS p ON rp.PRODUCT_ID = p.PRODUCT_ID
WHERE p.DATE = (SELECT MAX(DATE) FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS WHERE PRODUCT_ID = rp.PRODUCT_ID)

ORDER BY PRODUCT_ID, data_type;
