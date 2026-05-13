# Predicting Airbnb Occupancy and Pricing Using Listing and Neighborhood Demographic Data

<<<<<<< HEAD
**Authors:** Roman Shrestha and Tanish Pradhan Wong Ah Sui

## Repo Structure
```
proposal/
    proposal.pdf
progress_report/
    progress_report.ipynb
    progress_report.pdf
data/
    modeling.ipynb        # all data collection, cleaning, and prep code
    data_dictionary.xlsx  # variable descriptions
    sample_airbnb.csv     # representative sample of the full dataset
    listings/             # raw city-level Airbnb listings 
```

## Notes
- The processed dataset is too large for GitHub. `sample_airbnb.csv` is a representative sample of the actual dataset.
- A Census API key is required to run the data pipeline. Sign up at https://api.census.gov/data/key_signup.html and set it as an environment variable:
=======
Authors: Roman Shrestha and Tanish Pradhan Wong Ah Sui

## Repo layout
- `proposal/proposal.pdf`
- `data/data_preparation.ipynb` (data collection, geocoding, merge, cleaning)
- `data/airbnb_cleaned*.csv` (cleaned dataset, split because of GitHub LFS limits)
- `data/sample_airbnb.csv` (small sample of the cleaned dataset)
- `data/data_dictionary.xlsx` (variable descriptions)
- `data/listings/` (raw per-city Airbnb files)
- `progress_report/` (progress report PDF + notebook)
- `rough-draft/` (rough draft PDF)
- `final-draft/final_draft.ipynb` (full analysis notebook)
- `final-draft/final_draft.pdf` (final report)
- `figures/` (saved figures from the analysis notebook)
- `outputs/` (txt/csv results from the analysis notebook)

## Notes
- The cleaned dataset is split across two CSVs because of GitHub's file size limit. Concatenate them or load both.
- Running `data_preparation.ipynb` from scratch requires a Census API key. Sign up at https://api.census.gov/data/key_signup.html and set it as `CENSUS_API_KEY` in your environment.
- `final_draft.ipynb` reads the cleaned CSVs directly, so you do not need a Census API key to rerun the analysis.
>>>>>>> 8c031ec2052138f2fc67f1988c070d7b12079837
