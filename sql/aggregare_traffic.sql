-- Aggregates raw API gateway logs into 1-minute tumbling windows
-- Designed to reduce 50M+ daily rows into manageable time-series sequences for the LSTM

WITH CleanedLogs AS (
    SELECT 
        DATE_TRUNC('minute', timestamp) AS window_time,
        endpoint_id,
        latency_ms,
        status_code
    FROM raw_api_logs
    WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
)
SELECT 
    window_time,
    endpoint_id,
    COUNT(*) AS request_count,
    AVG(latency_ms) AS avg_latency,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency,
    SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) AS error_count
FROM CleanedLogs
GROUP BY window_time, endpoint_id
ORDER BY window_time ASC;