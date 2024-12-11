#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np

# Define the number of days for each month
days_in_month = {
    'January': 31,
    'February': 28,
    'March': 31,
    'April': 30,
    'May': 31,
    'June': 30,
    'July': 31,
    'August': 31,
    'September': 30,
    'October': 31,
    'November': 30,
    'December': 31
}

# Create date ranges for each month
date_ranges = {month: pd.date_range(start=f'{month} 1, 2021', periods=days, freq='D') for month, days in days_in_month.items()}

# given previous data from npci to generate Data for January to December 2021
data = [
    {'Month': 'January', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 2782.42, 'Value (Cr) by Customers': 443725.01},
    {'Month': 'January', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 3802.67, 'Value (Cr) by Customers': 651108.51},
    {'Month': 'January', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1076.75, 'Value (Cr) by Customers': 119647.31},
    {'Month': 'February', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 2639.55, 'Value (Cr) by Customers': 424442.48},
    {'Month': 'February', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 3519.12, 'Value (Cr) by Customers': 620121.97},
    {'Month': 'February', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1027.97, 'Value (Cr) by Customers': 116864.36},
    {'Month': 'March', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3054.44, 'Value (Cr) by Customers': 482896.69},
    {'Month': 'March', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4076.32, 'Value (Cr) by Customers': 707410.47},
    {'Month': 'March', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1169.63, 'Value (Cr) by Customers': 133167.49},
    {'Month': 'April', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3109.24, 'Value (Cr) by Customers': 474049.90},
    {'Month': 'April', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4216.99, 'Value (Cr) by Customers': 710178.44},
    {'Month': 'April', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1184.38, 'Value (Cr) by Customers': 135343.96},
    {'Month': 'May', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3274.69, 'Value (Cr) by Customers': 497748.17},
    {'Month': 'May', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4499.20, 'Value (Cr) by Customers': 757276.68},
    {'Month': 'May', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1229.38, 'Value (Cr) by Customers': 142291.76},
    {'Month': 'June', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3266.80, 'Value (Cr) by Customers': 494088.00},
    {'Month': 'June', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4461.40, 'Value (Cr) by Customers': 745552.20},
    {'Month': 'June', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1197.30, 'Value (Cr) by Customers': 139950.10},
    {'Month': 'July', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3494.64, 'Value (Cr) by Customers': 518095.10},
    {'Month': 'July', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4743.66, 'Value (Cr) by Customers': 761247.87},
    {'Month': 'July', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1278.96, 'Value (Cr) by Customers': 147723.30},
    {'Month': 'August', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3767.68, 'Value (Cr) by Customers': 538113.42},
    {'Month': 'August', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4987.33, 'Value (Cr) by Customers': 775265.57},
    {'Month': 'August', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1363.05, 'Value (Cr) by Customers': 153577.58},
    {'Month': 'September', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 3780.89, 'Value (Cr) by Customers': 541624.06},
    {'Month': 'September', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 4972.81, 'Value (Cr) by Customers': 774845.98},
    {'Month': 'September', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1333.38, 'Value (Cr) by Customers': 153708.60},
    {'Month': 'October', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 4139.92, 'Value (Cr) by Customers': 589299.18},
    {'Month': 'October', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 5337.96, 'Value (Cr) by Customers': 838371.70},
    {'Month': 'October', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1417.81, 'Value (Cr) by Customers': 167819.30},
    {'Month': 'November', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 4099.08, 'Value (Cr) by Customers': 599748.64},
    {'Month': 'November', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 5281.68, 'Value (Cr) by Customers': 854939.06},
    {'Month': 'November', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1360.45, 'Value (Cr) by Customers': 166257.62},
    {'Month': 'December', 'UPI Banks': 'Google Pay', 'Volume (Mn) By Customers': 4375.29, 'Value (Cr) by Customers': 629285.36},
    {'Month': 'December', 'UPI Banks': 'PhonePe', 'Volume (Mn) By Customers': 5642.66, 'Value (Cr) by Customers': 901006.65},
    {'Month': 'December', 'UPI Banks': 'Paytm Payments Bank App', 'Volume (Mn) By Customers': 1442.58, 'Value (Cr) by Customers': 165694.96}
]

# Initialize an empty DataFrame to store the data
expanded_df = pd.DataFrame(columns=['Date', 'UPI Banks', 'Volume (Mn) By Customers', 'Value (Cr) by Customers', 'Month', 'Year'])

# Iterate over the data and expand it for each day in the respective month
for entry in data:
    month = entry['Month']
    days = date_ranges[month]
    num_days = len(days)
    
    volume_values = np.random.normal(entry['Volume (Mn) BSy Customers'], scale=0.1*entry['Volume (Mn) By Customers'], size=num_days)
    value_values = np.random.normal(entry['Value (Cr) by Customers'], scale=0.1*entry['Value (Cr) by Customers'], size=num_days)
    
    repeated_rows = pd.concat([pd.DataFrame({'Date': days, 'UPI Banks': np.repeat(entry['UPI Banks'], num_days),
                                             'Volume (Mn) By Customers': volume_values,
                                             'Value (Cr) by Customers': value_values,
                                             'Month': np.repeat(month, num_days),
                                             'Year': np.repeat(2023, num_days)})], ignore_index=True)
    
    expanded_df = pd.concat([expanded_df, repeated_rows], ignore_index=True)

# Save the data to a CSV file
expanded_df.to_csv('jan_to_dec_2021_.csv', index=False)
print("Data saved to 'jan_to_dec_2021.csv'")

