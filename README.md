# Predictive Maintenance – Machine Failure Prediction

Predicts whether an industrial milling machine will fail, using sensor readings. Built with XGBoost and served through a FastAPI REST API.

## Dataset
- **AI4I 2020 Predictive Maintenance Dataset** (UCI Machine Learning Repository)
- 10,000 records, 14 columns, no missing values
- Synthetic dataset that reflects real industrial data
- Highly imbalanced: only **3.4% failures** (339 of 10,000)

## Approach
1. **Cleaning:** Dropped ID columns and failure-mode columns (TWF, HDF, PWF, OSF, RNF) to prevent **data leakage**.
2. **Feature engineering:** Added physics-based features:
   - `temp_diff` = process temp − air temp
   - `power` = torque × angular speed (W)
   - `strain` = tool wear × torque
3. **Model:** XGBoost with `scale_pos_weight` to handle class imbalance.
4. **Validation:** Stratified 80/20 train/test split and 5-fold cross-validation.
5. **Threshold tuning:** Chose the decision threshold (0.8) using out-of-fold predictions on training data only, so the test set stayed unseen.

## Results (test set, failure class)
| Metric | Score |
|---|---|
| F1-score | 0.87 |
| Precision | 0.93 |
| Recall | 0.81 |
| 5-fold CV Mean F1 | [CV score] |

Accuracy (99.2%) is **not** used as the main metric. A model that always predicts "no failure" would still reach ~96.6% accuracy.

![Feature Importance](feature_importance.png)

## Run the API
```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Open `http://127.0.0.1:8000/docs`.

### Example request – `POST /predict`
```json
{"type": "L", "air_temp": 299, "process_temp": 309, "rpm": 1400, "torque": 60, "tool_wear": 220}
```
### Example response
```json
{"failure": 1, "probability": 0.97}
```
`type`: product quality, `L` (low), `M` (medium) or `H` (high). Temperatures are in Kelvin.

## Project Structure
```
├── training.ipynb          # Data cleaning, training, evaluation
├── main.py                 # FastAPI app
├── model.pkl               # Trained XGBoost model
├── config.json             # Threshold + feature order
├── feature_importance.png
└── requirements.txt
```

## Limitations
- The dataset is synthetic and its failures follow fixed rules, so real-world scores would likely be lower.
- There is no time-series information. Real predictive maintenance often tracks degradation over time.
- 13 of 68 test failures were missed. Lowering the threshold would raise recall at the cost of more false alarms.

## Tech Stack
Python · Pandas · Scikit-learn · XGBoost · FastAPI · Uvicorn