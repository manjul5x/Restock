-- =================================================================================
-- SNOWFLAKE VALIDATION QUERIES (FIXED - UPPERCASE COLUMNS)
-- =================================================================================
-- Run these queries in Snowflake UI to get actual data for validation
-- Copy and paste the results to validate data transformation and forecast quality
-- =================================================================================

-- =================================================================================
-- QUERY 1: RAW DATA OVERVIEW (STAGE Schema)
-- =================================================================================
-- Purpose: Get overview of raw data in STAGE schema
-- Run this first to understand the raw data structure and content

SELECT 
    'OUTFLOW_BHASIN' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products,
    COUNT(DISTINCT LOCATION_ID) as unique_locations,
    MIN(DATE) as earliest_date,
    MAX(DATE) as latest_date,
    COUNT(DISTINCT DATE) as unique_dates
FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN

UNION ALL

SELECT 
    'PRODUCT_MASTER_BHASIN' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products,
    NULL as unique_locations,
    NULL as earliest_date,
    NULL as latest_date,
    NULL as unique_dates
FROM RESTOCK_DB.STAGE.PRODUCT_MASTER_BHASIN;

-- =================================================================================
-- QUERY 2: RAW DATA SAMPLE (STAGE Schema)
-- =================================================================================
-- Purpose: Get sample raw data to see actual content
-- This shows what the raw data looks like

SELECT 
    DATE,
    PRODUCT_ID,
    LOCATION_ID,
    DEMAND,
    UNIT_PRICE,
    STOCK_LEVEL,
    INCOMING_INVENTORY,
    PRODUCT_CATEGORY
FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN
ORDER BY DATE DESC, PRODUCT_ID
LIMIT 10;

-- =================================================================================
-- QUERY 3: PROCESSED DATA OVERVIEW (TRANSFORMATION Schema)
-- =================================================================================
-- Purpose: Get overview of processed data with regressors
-- This shows how the raw data was enhanced

SELECT 
    'PROCESSED_DATA_WITH_REGRESSORS' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products,
    COUNT(DISTINCT LOCATION_ID) as unique_locations,
    MIN(DATE) as earliest_date,
    MAX(DATE) as latest_date,
    COUNT(DISTINCT DATE) as unique_dates
FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS;

-- =================================================================================
-- QUERY 4: PROCESSED DATA SAMPLE (TRANSFORMATION Schema)
-- =================================================================================
-- Purpose: Get sample processed data to see regressor features
-- This shows the enhanced data with new features

SELECT 
    DATE,
    PRODUCT_ID,
    LOCATION_ID,
    DEMAND,
    RP_LAG_1,
    RP_LAG_2, 
    RP_LAG_3,
    HALF_RP_LAG_1,
    HALF_RP_LAG_2,
    SEASON_1,
    SEASON_2,
    SEASON_3,
    WEEK_1,
    WEEK_2,
    WEEK_3,
    RECENCY_1,
    RECENCY_2,
    RECENCY_3
FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS
ORDER BY DATE DESC, PRODUCT_ID
LIMIT 10;

-- =================================================================================
-- QUERY 5: FORECAST DATA OVERVIEW (TRANSFORMATION Schema)
-- =================================================================================
-- Purpose: Get overview of forecast data
-- This shows the final predictions

SELECT 
    'FUTURE_PREDICTIONS_RESULTS' as table_name,
    COUNT(*) as total_predictions,
    COUNT(DISTINCT PRODUCT_ID) as unique_products,
    COUNT(DISTINCT LOCATION_ID) as unique_locations,
    MIN(FORECASTED_ON) as earliest_forecast,
    MAX(FORECASTED_ON) as latest_forecast,
    COUNT(DISTINCT FORECASTED_ON) as unique_forecast_dates
FROM RESTOCK_DB.TRANSFORMATION.FUTURE_PREDICTIONS_RESULTS;

-- =================================================================================
-- QUERY 6: FORECAST DATA SAMPLE (TRANSFORMATION Schema)
-- =================================================================================
-- Purpose: Get sample forecast data to see actual predictions
-- This shows the final forecast values

SELECT 
    PRODUCT_ID,
    LOCATION_ID,
    FORECASTED_ON,
    PREDICTED_OUTFLOW_TOTAL,
    PREDICTED_OUTFLOW_DAILY,
    RISK_PERIOD_DAYS,
    CREATED_AT
FROM RESTOCK_DB.TRANSFORMATION.FUTURE_PREDICTIONS_RESULTS
ORDER BY FORECASTED_ON DESC, PRODUCT_ID
LIMIT 10;

-- =================================================================================
-- QUERY 7: DATA TRANSFORMATION VERIFICATION
-- =================================================================================
-- Purpose: Verify that raw data was properly transformed
-- This compares raw vs processed data for the same products

WITH raw_data AS (
    SELECT 
        PRODUCT_ID,
        COUNT(*) as raw_records,
        MIN(DATE) as raw_start_date,
        MAX(DATE) as raw_end_date
    FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN
    GROUP BY PRODUCT_ID
),
processed_data AS (
    SELECT 
        PRODUCT_ID,
        COUNT(*) as processed_records,
        MIN(DATE) as processed_start_date,
        MAX(DATE) as processed_end_date
    FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS
    GROUP BY PRODUCT_ID
)
SELECT 
    r.PRODUCT_ID,
    r.raw_records,
    p.processed_records,
    CASE 
        WHEN r.raw_records = p.processed_records THEN '✅ MATCH'
        ELSE '❌ MISMATCH'
    END as record_count_match,
    r.raw_start_date,
    p.processed_start_date,
    r.raw_end_date,
    p.processed_end_date
FROM raw_data r
JOIN processed_data p ON r.PRODUCT_ID = p.PRODUCT_ID
ORDER BY r.PRODUCT_ID
LIMIT 10;

-- =================================================================================
-- QUERY 8: FORECAST QUALITY VERIFICATION
-- =================================================================================
-- Purpose: Verify forecast quality and reasonableness
-- This checks if forecasts make sense

SELECT 
    PRODUCT_ID,
    LOCATION_ID,
    PREDICTED_OUTFLOW_TOTAL,
    PREDICTED_OUTFLOW_DAILY,
    RISK_PERIOD_DAYS,
    CASE 
        WHEN PREDICTED_OUTFLOW_TOTAL > 0 AND PREDICTED_OUTFLOW_DAILY > 0 THEN '✅ VALID'
        WHEN PREDICTED_OUTFLOW_TOTAL = 0 AND PREDICTED_OUTFLOW_DAILY = 0 THEN '⚠️  ZERO'
        ELSE '❌ INVALID'
    END as forecast_quality,
    FORECASTED_ON
FROM RESTOCK_DB.TRANSFORMATION.FUTURE_PREDICTIONS_RESULTS
ORDER BY PREDICTED_OUTFLOW_TOTAL DESC
LIMIT 10;

-- =================================================================================
-- QUERY 9: REGRESSOR FEATURE VERIFICATION
-- =================================================================================
-- Purpose: Verify that regressor features were properly computed
-- This checks if the new features have reasonable values

SELECT 
    PRODUCT_ID,
    DATE,
    DEMAND,
    RP_LAG_1,
    RP_LAG_2,
    RP_LAG_3,
    HALF_RP_LAG_1,
    HALF_RP_LAG_2,
    SEASON_1,
    SEASON_2,
    SEASON_3,
    WEEK_1,
    WEEK_2,
    WEEK_3,
    RECENCY_1,
    RECENCY_2,
    RECENCY_3
FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS
WHERE PRODUCT_ID IN (
    SELECT DISTINCT PRODUCT_ID 
    FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS 
    LIMIT 3
)
ORDER BY PRODUCT_ID, DATE DESC
LIMIT 15;

-- =================================================================================
-- QUERY 10: COMPLETE PIPELINE SUMMARY
-- =================================================================================
-- Purpose: Get complete summary of the pipeline results
-- This provides the final validation summary

SELECT 
    'PIPELINE SUMMARY' as summary_type,
    'STAGE' as schema_name,
    'OUTFLOW_BHASIN' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products
FROM RESTOCK_DB.STAGE.OUTFLOW_BHASIN

UNION ALL

SELECT 
    'PIPELINE SUMMARY' as summary_type,
    'STAGE' as schema_name,
    'PRODUCT_MASTER_BHASIN' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products
FROM RESTOCK_DB.STAGE.PRODUCT_MASTER_BHASIN

UNION ALL

SELECT 
    'PIPELINE SUMMARY' as summary_type,
    'TRANSFORMATION' as schema_name,
    'PROCESSED_DATA_WITH_REGRESSORS' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products
FROM RESTOCK_DB.TRANSFORMATION.PROCESSED_DATA_WITH_REGRESSORS

UNION ALL

SELECT 
    'PIPELINE SUMMARY' as summary_type,
    'TRANSFORMATION' as schema_name,
    'FUTURE_PREDICTIONS_RESULTS' as table_name,
    COUNT(*) as total_rows,
    COUNT(DISTINCT PRODUCT_ID) as unique_products
FROM RESTOCK_DB.TRANSFORMATION.FUTURE_PREDICTIONS_RESULTS;

-- =================================================================================
-- INSTRUCTIONS FOR RUNNING THESE QUERIES
-- =================================================================================
-- 1. Copy each query individually
-- 2. Paste into Snowflake UI query editor
-- 3. Run each query
-- 4. Copy the results and share them
-- 5. I will analyze the actual data to validate:
--    - Raw data quality and structure
--    - Data transformation accuracy
--    - Regressor feature computation
--    - Forecast quality and reasonableness
--    - Complete pipeline validation
-- =================================================================================
