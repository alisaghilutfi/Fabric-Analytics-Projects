CREATE TABLE [Gold].[Dim_Date] (

	[date_key] int NOT NULL, 
	[full_date] date NOT NULL, 
	[year] int NULL, 
	[quarter] int NULL, 
	[month] int NULL, 
	[month_name] varchar(15) NULL, 
	[day] int NULL, 
	[day_of_week] int NULL, 
	[day_name] varchar(15) NULL, 
	[is_weekend] bit NULL
);