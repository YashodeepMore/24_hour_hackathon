# MicroFinance Hub

A web application designed to transform financial access for micro-entrepreneurs in emerging markets through alternative credit scoring and business intelligence.

## Features

- User authentication and profile management
- Alternative credit scoring using non-traditional data points
- Real-time expense tracking and categorization
- Cash flow prediction and visualization
- Interactive financial education modules
- Business intelligence dashboard

## Technical Stack

- Backend: Flask (Python)
- Database: SQLite
- Frontend: HTML, CSS, JavaScript, Bootstrap 5
- Machine Learning: scikit-learn
- Data Visualization: Chart.js

## Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd microfinance-hub
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize the database:
```bash
flask db init
flask db migrate
flask db upgrade
```

5. Run the application:
```bash
python app.py
```

6. Access the application at `http://localhost:5000`

## Project Structure

```
microfinance-hub/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static files (CSS, JS, images)
│   └── css/
│       └── style.css
├── templates/         # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── login.html
│   ├── register.html
│   ├── cash_flow.html
│   └── financial_education.html
└── microfinance.db    # SQLite database
```

## Key Features Implementation

### Alternative Credit Scoring
The application uses a Random Forest Classifier to generate credit scores based on:
- Monthly income
- Expense ratio
- Savings rate
- Transaction frequency

### Cash Flow Prediction
- 30-day cash flow forecasting
- Interactive visualization using Chart.js
- Real-time updates based on transaction data

### Financial Education
- Interactive learning modules
- Progress tracking
- Gamified content delivery

## Security Considerations

- Password hashing using Werkzeug
- SQL injection prevention through SQLAlchemy
- CSRF protection
- Secure session management

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 