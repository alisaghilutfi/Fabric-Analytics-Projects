CREATE TABLE [Gold].[Fact_Sales] (

	[order_id] varchar(8000) NULL, 
	[product_id] varchar(8000) NULL, 
	[product_category_name] varchar(8000) NULL, 
	[customer_id] varchar(8000) NULL, 
	[seller_id] varchar(8000) NULL, 
	[order_purchase_timestamp] datetime2(6) NULL, 
	[order_date] date NULL, 
	[price] float NULL, 
	[freight_value] float NULL, 
	[total_order_value] float NULL
);