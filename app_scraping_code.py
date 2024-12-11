#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install app_store_scraper')


# In[39]:


from app_store_scraper import AppStore
import pandas as pd

# passing the MobilePayment app parameters to scrape application reviews from appstore website
Paytm = AppStore(country='in', app_name='paytm-secure-upi-payments', app_id='473941634')

# Fetching reviws for Paytm
Paytm.review(how_many=2000)

# Filter reviews for the year 2017-2024,
reviews_2024 = [review for review in Paytm.reviews if review['date'].year == 2024]
 
df_Paytm_reviews_2024 = pd.DataFrame(reviews_2024)

# Printing few reviews 
print(df_Paytm_reviews_2024)



# In[40]:


# Save DataFrame to CSVvof paytm reviews
df_Paytm_reviews_2024.to_csv('Paytm_2024.csv', index=False)
print("Reviews for the year 2024 have been saved to Paytm_reviews_2024.csv")


# In[3]:


from app_store_scraper import AppStore
import pandas as pd

# passing the MobilePayment app  parameters to scrape application reviews from appstore website.
googlepay = AppStore(country='in', app_name='google-pay-save-pay-manage', app_id='1193357041')

# Fetch reviews         
googlepay.review(how_many=2000)

# Access the reviews for googlepay
reviews = googlepay.reviews

# Filter reviews for the year 2017-2024
reviews_2024 = [review for review in reviews if review['date'].year == 2024]

# extract reviews into data frame 
df_googlepay_reviews_2024 = pd.DataFrame(reviews_2024)

# Print  the DataFrame
print(df_googlepay_reviews_2024)


# In[6]:


# Save DataFrame to CSV for googlepay reviews
df_googlepay_reviews_2024.to_csv('googlepay_2024.csv', index=False)
print("Reviews for the year 2024 have been saved to googlepay_2024.csv")


# In[5]:


from app_store_scraper import AppStore
import pandas as pd

# passing the MobilePayment app  parameters to scrape application reviews from appstore website.
Phonepe = AppStore(country='in', app_name='phonepe-secure-payments-app', app_id='1170055821')

# Fetch reviews for phone pe       
Phonepe.review(how_many=2000)

# Filter reviews for the year 2017-2024,
reviews_2024 = [review for review in Phonepe.reviews if review['date'].year == 2024]

# extract reviews into data frame 
df_Phonepe_reviews_2024 = pd.DataFrame(reviews_2024)

# Print or further process the DataFrame
print(df_Phonepe_reviews_2024)


# In[7]:


# Save DataFrame to CSV for Phonepe reviews
df_googlepay_reviews_2024.to_csv('Phonepe_reviews_2024', index=False)
print("Reviews for the year 2024 have been saved to Phonepe_reviews_2024")

