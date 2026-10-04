CREATE TABLE [Gold].[Agg_Customer_Intelligence] (

	[customer_unique_id] varchar(8000) NULL, 
	[lifetime_value] float NULL, 
	[total_orders] int NULL, 
	[avg_order_value] float NULL, 
	[last_purchase_date] datetime2(6) NULL, 
	[first_purchase_date] datetime2(6) NULL, 
	[customer_tenure_days] int NULL
);