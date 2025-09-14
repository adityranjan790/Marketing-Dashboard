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

<img width="1058" height="1759" alt="image" src="https://github.com/user-attachments/assets/81857ebb-de31-4396-a656-666ec5c1b558" />
<img width="1891" height="915" alt="image" src="https://github.com/user-attachments/assets/7cba1e10-151f-476b-9866-dc1e30bb6d8b" />
<img width="1886" height="921" alt="image" src="https://github.com/user-attachments/assets/bbfa8a91-b4b5-4fd4-8351-61011cfacbff" />
<img width="1851" height="764" alt="image" src="https://github.com/user-attachments/assets/74b307f0-4cf5-46e0-9a5d-640e68b5c7c3" />
<img width="941" height="1109" alt="image" src="https://github.com/user-attachments/assets/af8cab79-bad5-403f-9ddd-99f19623029e" />
<img width="1813" height="924" alt="image" src="https://github.com/user-attachments/assets/ee9749eb-aaed-4b28-8300-a4c4aa3c9393" />
