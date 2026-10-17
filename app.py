import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the Linear Regression model trained with scikit-learn 1.9.1[cite: 4]
with open('linear.pkl', 'rb') as f:
  model = pickle.load(f)

# HTML Layout and CSS Style embedded directly inside app.py
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Cost & Charge Predictor</title>
    <!-- Bootstrap 5 CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .card {
            border: none;
            border-radius: 1rem;
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }
        .btn-primary {
            background-color: #4e73df;
            border: none;
            transition: background-color 0.2s;
        }
        .btn-primary:hover {
            background-color: #2e59d9;
        }
        .form-label {
            color: #495057;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container py-5">
        <div class="row justify-content-center">
            <div class="col-lg-7">
                <div class="card p-4 p-md-5 bg-white">
                    <div class="text-center mb-4">
                        <h2 class="fw-bold text-primary">Insurance Cost Predictor</h2>
                        <p class="text-muted">Enter the details below to predict expected medical charges.</p>
                    </div>

                    <form action="/predict" method="POST">
                        <div class="row g-3">
                            <!-- age -->
                            <div class="col-md-6">
                                <label class="form-label">Age</label>
                                <input type="number" class="form-control" name="age" placeholder="e.g. 30" required>
                            </div>

                            <!-- sex -->
                            <div class="col-md-6">
                                <label class="form-label">Sex</label>
                                <select class="form-select" name="sex" required>
                                    <option value="female">Female</option>
                                    <option value="male">Male</option>
                                </select>
                            </div>

                            <!-- bmi -->
                            <div class="col-md-6">
                                <label class="form-label">BMI</label>
                                <input type="number" step="0.1" class="form-control" name="bmi" placeholder="e.g. 25.4" required>
                            </div>

                            <!-- children -->
                            <div class="col-md-6">
                                <label class="form-label">Children</label>
                                <input type="number" class="form-control" name="children" placeholder="e.g. 1" required>
                            </div>

                            <!-- smoker -->
                            <div class="col-md-6">
                                <label class="form-label">Smoker</label>
                                <select class="form-select" name="smoker" required>
                                    <option value="no">No</option>
                                    <option value="yes">Yes</option>
                                </select>
                            </div>

                            <!-- region -->
                            <div class="col-md-6">
                                <label class="form-label">Region</label>
                                <select class="form-select" name="region" required>
                                    <option value="northeast">Northeast</option>
                                    <option value="northwest">Northwest</option>
                                    <option value="southeast">Southeast</option>
                                    <option value="southwest">Southwest</option>
                                </select>
                            </div>
                        </div>

                        <div class="d-grid mt-4">
                            <button type="submit" class="btn btn-primary btn-lg shadow-sm">Predict Charges</button>
                        </div>
                    </form>

                    {% if prediction_text %}
                    <div class="alert alert-success mt-4 text-center shadow-sm" role="alert">
                        <h4 class="alert-heading mb-1 fw-bold">{{ prediction_text }}</h4>
                    </div>
                    {% endif %}

                    {% if error_text %}
                    <div class="alert alert-danger mt-4 text-center shadow-sm" role="alert">
                        <p class="mb-0">{{ error_text }}</p>
                    </div>
                    {% endif %}
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


@app.route('/')
def home():
  return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
  try:
    # 1. Retrieve the 6 required features from form input[cite: 4]
    age = float(request.form['age'])
    sex_str = request.form['sex']
    bmi = float(request.form['bmi'])
    children = float(request.form['children'])
    smoker_str = request.form['smoker']
    region_str = request.form['region']

    # 2. Map categorical fields to standard numerical equivalents (Alphabetical LabelEncoder mappings)
    sex_map = {'female': 0, 'male': 1}
    smoker_map = {'no': 0, 'yes': 1}
    region_map = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}

    sex = sex_map.get(sex_str, 0)
    smoker = smoker_map.get(smoker_str, 0)
    region = region_map.get(region_str, 0)

    # 3. Create feature array matching the exact feature order[cite: 4]
    # Order: age, sex, bmi, children, smoker, region[cite: 4]
    features = [[age, sex, bmi, children, smoker, region]]

    # 4. Make prediction
    prediction = model.predict(features)[0]

    return render_template_string(
        HTML_TEMPLATE,
        prediction_text=f'Estimated Medical Cost: ${prediction:,.2f}',
    )

  except Exception as e:
    return render_template_string(
        HTML_TEMPLATE, error_text=f'Error in processing: {str(e)}'
    )


if __name__ == '__main__':
  app.run(debug=True)
