# Marketing Intelligence Dashboard

This is a full example BI dashboard (Streamlit) that links campaign-level marketing data to business outcomes.
It includes synthetic example data for Facebook, Google, TikTok and Business performance for 120 days.

## What is included
- `app.py` : Streamlit application
- `data/` : synthetic CSV files (Facebook.csv, Google.csv, TikTok.csv, Business.csv)
- `requirements.txt`

## How to run locally
1. Install Python 3.8+ and pip.
2. Create a virtual env (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Mac / Linux
   venv\Scripts\activate    # Windows PowerShell
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run app:
   ```bash
   streamlit run app.py
   ```
5. The app will open at `http://localhost:8501` by default.

## How to host
- **Streamlit Cloud**: Create a GitHub repo with these files and deploy.
- **Heroku / Render / Fly**: Containerize or use their Python app deployment flow.

## Notes for your submission / assessment
- Replace synthetic CSVs in `data/` with real exported CSVs (Facebook.csv, Google.csv, TikTok.csv, Business.csv) keeping the same column names.
- Adjust filters, date parsing, and derived metrics as needed for real data quirks (timezones, duplicates, attribution windows).
- Add authentication or access controls if required by your placement project rules.

---
Generated automatically as an example project for the provided assignment.
