# Predicting Airbnb Occupancy and Pricing Using Listing and Neighborhood Demographic Data

**Authors:** Roman Shrestha and Tanish Pradhan Wong Ah Sui

## Repo Structure
```
proposal/
    proposal.pdf
data/
    modeling.ipynb        # all data collection, cleaning, and prep code
    data_dictionary.xlsx  # variable descriptions
    sample_airbnb.csv     # representative sample of the full dataset
    listings/             # raw city-level Airbnb listings 
```

## Notes
- The processed dataset is too large for GitHub. `sample_airbnb.csv` is a representative sample of the actual dataset.
- A Census API key is required to run the data pipeline. Sign up at https://api.census.gov/data/key_signup.html and set it as an environment variable:
