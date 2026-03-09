CREATE SCHEMA IF NOT EXISTS data;

CREATE OR REPLACE VIEW data.v_firm_monthly AS
SELECT * FROM data.agg_firm_monthly;

CREATE OR REPLACE VIEW data.v_channel_monthly AS
SELECT * FROM data.agg_channel_monthly;

CREATE OR REPLACE VIEW data.v_ticker_monthly AS
SELECT * FROM data.agg_ticker_monthly;

CREATE OR REPLACE VIEW data.v_geo_monthly AS
SELECT * FROM data.agg_geo_monthly;

CREATE OR REPLACE VIEW data.v_segment_monthly AS
SELECT * FROM data.agg_segment_monthly;
