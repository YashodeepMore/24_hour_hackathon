from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import random
from werkzeug.security import generate_password_hash, check_password_hash
from credit_scorer import CreditScorer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///microfinance.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Add custom Jinja2 filter for absolute value
@app.template_filter('abs')
def abs_filter(value):
    return abs(value)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Simple password hashing
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    is_investor = db.Column(db.Boolean, default=False)
    
    # Business metrics fields (will be updated after registration)
    business_type = db.Column(db.String(50))
    monthly_income = db.Column(db.Float, default=0.0)
    business_id = db.Column(db.String(50))
    annual_revenue = db.Column(db.Float, default=0.0)
    net_profit_margin = db.Column(db.Float, default=0.0)
    total_assets = db.Column(db.Float, default=0.0)
    total_liabilities = db.Column(db.Float, default=0.0)
    debt_equity_ratio = db.Column(db.Float, default=0.0)
    cash_flow = db.Column(db.Float, default=0.0)
    years_active = db.Column(db.Integer, default=0)
    industry = db.Column(db.String(100))
    num_employees = db.Column(db.Integer, default=0)
    credit_history_length = db.Column(db.Integer, default=0)
    late_payments = db.Column(db.Integer, default=0)
    credit_utilization = db.Column(db.Float, default=0.0)
    bankruptcy_flag = db.Column(db.Boolean, default=False)
    
    # Investor-specific fields
    investment_preference = db.Column(db.String(20))
    max_investment_amount = db.Column(db.Float, default=0.0)
    
    # Relationships
    transactions = db.relationship('Transaction', backref='user', lazy=True)
    investments_made = db.relationship('Investment', 
                                     foreign_keys='Investment.investor_id',
                                     backref='investor',
                                     lazy=True)
    investments_received = db.relationship('Investment',
                                         foreign_keys='Investment.business_id',
                                         backref='business',
                                         lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Expense(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    amount = db.Column(db.Float, nullable=False)
    type = db.Column(db.String(10), nullable=False)  # 'income' or 'expense'
    category = db.Column(db.String(50), nullable=False)  # New field for transaction category
    description = db.Column(db.String(200))
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Investment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    investor_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    business_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    status = db.Column(db.String(20), nullable=False, default='active')  # active, completed, withdrawn
    return_rate = db.Column(db.Float, nullable=False, default=0.0)
    risk_score = db.Column(db.Float, nullable=False, default=0.0)

def generate_training_data():
    """Generate synthetic business data for training the credit scoring model."""
    # Define industry types
    industries = ['Retail', 'Food & Beverage', 'Service', 'Manufacturing', 'Technology', 'Healthcare']
    
    # Generate 1000 sample businesses
    data = []
    for i in range(1000):
        # Generate random business metrics
        years_active = random.randint(1, 20)
        num_employees = random.randint(1, 100)
        annual_revenue = random.uniform(50000, 5000000)
        net_profit_margin = random.uniform(0.05, 0.25)  # 5% to 25%
        total_assets = annual_revenue * random.uniform(0.5, 2.0)
        total_liabilities = total_assets * random.uniform(0.1, 0.8)
        cash_flow = annual_revenue * random.uniform(0.05, 0.2)
        credit_history_length = random.randint(1, years_active)
        late_payments = random.randint(0, 5)
        credit_utilization = random.uniform(0.1, 0.9)  # 10% to 90%
        bankruptcy_flag = random.random() < 0.05  # 5% chance of bankruptcy
        
        # Calculate debt-equity ratio
        if total_liabilities > 0:
            debt_equity_ratio = total_liabilities / (total_assets - total_liabilities)
        else:
            debt_equity_ratio = 0.0
            
        # Determine credit score (1 for good, 0 for bad)
        # Factors that positively influence credit score:
        # - Higher annual revenue
        # - Higher net profit margin
        # - Lower debt-equity ratio
        # - Lower late payments
        # - Lower credit utilization
        # - No bankruptcy
        # - Longer credit history
        score = 0
        if (annual_revenue > 1000000 and  # High revenue
            net_profit_margin > 0.15 and  # Good profit margin
            debt_equity_ratio < 1.0 and  # Reasonable debt
            late_payments <= 1 and  # Few late payments
            credit_utilization < 0.5 and  # Low credit utilization
            not bankruptcy_flag and  # No bankruptcy
            credit_history_length > 5):  # Long credit history
            score = 1
            
        data.append({
            'years_active': years_active,
            'num_employees': num_employees,
            'annual_revenue': annual_revenue,
            'net_profit_margin': net_profit_margin,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'debt_equity_ratio': debt_equity_ratio,
            'cash_flow': cash_flow,
            'credit_history_length': credit_history_length,
            'late_payments': late_payments,
            'credit_utilization': credit_utilization,
            'bankruptcy_flag': bankruptcy_flag,
            'credit_score': score
        })
    
    return pd.DataFrame(data)

# Initialize credit scorer
credit_scorer = CreditScorer()

# Load and train the model with the dataset
try:
    df = pd.read_csv('business_credit_score_dataset.csv')
    training_results = credit_scorer.train(df)
    print("Model training results:", training_results)
except Exception as e:
    print("Error loading or training the model:", str(e))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        user_type = request.form['user_type']

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register'))

        user = User(
            username=username,
            email=email,
            is_investor=(user_type == 'investor')
        )

        if user_type == 'business':
            user.business_type = request.form['business_type']
            user.industry = request.form['industry']
        else:  # investor
            user.investment_preference = request.form['investment_preference']
            user.max_investment_amount = float(request.form['investment_amount'])

        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/setup_profile', methods=['GET', 'POST'])
@login_required
def setup_profile():
    user = User.query.get(session['user_id'])
    
    if request.method == 'POST':
        # Update user with business information
        user.business_id = request.form['business_id']
        user.business_type = request.form['business_type']
        user.industry = request.form['industry']
        user.years_active = int(request.form['years_active'])
        user.num_employees = int(request.form['num_employees'])
        user.annual_revenue = float(request.form['annual_revenue'])
        user.net_profit_margin = float(request.form['net_profit_margin'])
        user.total_assets = float(request.form['total_assets'])
        user.total_liabilities = float(request.form['total_liabilities'])
        user.cash_flow = float(request.form['cash_flow'])
        user.credit_history_length = int(request.form['credit_history_length'])
        user.late_payments = int(request.form['late_payments'])
        user.credit_utilization = float(request.form['credit_utilization'])
        user.bankruptcy_flag = request.form['bankruptcy_flag'] == 'true'
        
        # Calculate debt-equity ratio
        if user.total_liabilities > 0:
            user.debt_equity_ratio = user.total_liabilities / (user.total_assets - user.total_liabilities)
        else:
            user.debt_equity_ratio = 0.0
        
        db.session.commit()
        flash('Profile setup complete!')
        return redirect(url_for('dashboard'))
    
    return render_template('setup_profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user_type = request.form['user_type']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            if (user_type == 'business' and not user.is_investor) or \
               (user_type == 'investor' and user.is_investor):
                session['user_id'] = user.id
                if user.is_investor:
                    return redirect(url_for('investor_dashboard'))
                else:
                    return redirect(url_for('dashboard'))
            else:
                flash('Invalid user type for this account.', 'danger')
        else:
            flash('Invalid username or password', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    user = User.query.get(session['user_id'])
    
    # Check if user has completed profile setup
    if not user.business_id:
        return redirect(url_for('setup_profile'))
    
    # Get user's transactions
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.date.desc()).limit(5).all()
    
    # Calculate cash flow metrics
    cash_flow_metrics = calculate_cash_flow_metrics(transactions)
    
    # Prepare data for credit scoring
    business_data = {
        'Annual_Revenue': user.annual_revenue,
        'Net_Profit_Margin': user.net_profit_margin,
        'Total_Assets': user.total_assets,
        'Total_Liabilities': user.total_liabilities,
        'Debt_Equity_Ratio': user.debt_equity_ratio,
        'Cash_Flow': cash_flow_metrics['annual_cash_flow'],
        'Years_Active': user.years_active,
        'Industry': user.industry,
        'Num_Employees': user.num_employees,
        'Credit_History_Length': user.credit_history_length,
        'Late_Payments': user.late_payments,
        'Credit_Utilization': user.credit_utilization,
        'Bankruptcy_Flag': user.bankruptcy_flag
    }
    
    # Get credit score prediction
    try:
        credit_score = credit_scorer.predict(business_data)
    except Exception as e:
        print("Error predicting credit score:", str(e))
        credit_score = 600  # Default score if prediction fails
    
    # Calculate risk score from credit score
    risk_score = calculate_risk_score(user)
    
    return render_template('dashboard.html', 
                         user=user,
                         transactions=transactions,
                         total_expenses=cash_flow_metrics['monthly_avg_expense'] * 12,
                         total_income=cash_flow_metrics['monthly_avg_income'] * 12,
                         credit_score=credit_score,
                         risk_score=risk_score,
                         cash_flow_metrics=cash_flow_metrics)

@app.route('/add_transaction', methods=['POST'])
@login_required
def add_transaction():
    try:
        amount = float(request.form['amount'])
        type = request.form['type']
        category = request.form['category']
        description = request.form.get('description', '')
        
        if not category:
            flash('Please select a category')
            return redirect(url_for('analytics'))
        
        transaction = Transaction(
            amount=amount,
            type=type,
            category=category,
            description=description,
            date=datetime.utcnow(),
            user_id=session['user_id']
        )
        
        db.session.add(transaction)
        db.session.commit()
        
        flash('Transaction added successfully!')
        return redirect(url_for('analytics'))
    except (KeyError, ValueError) as e:
        flash('Error adding transaction. Please check all fields are filled correctly.')
        return redirect(url_for('analytics'))

@app.route('/transactions')
@login_required
def transactions():
    user = User.query.get(session['user_id'])
    transactions = Transaction.query.filter_by(user_id=user.id).order_by(Transaction.date.desc()).all()
    return render_template('transactions.html', transactions=transactions)

@app.route('/analytics')
@login_required
def analytics():
    user = User.query.get(session['user_id'])
    
    # Get transactions for the last 6 months
    six_months_ago = datetime.utcnow() - timedelta(days=180)
    transactions = Transaction.query.filter(
        Transaction.user_id == user.id,
        Transaction.date >= six_months_ago
    ).all()
    
    # Prepare data for charts
    monthly_data = {}
    category_data = {'income': {}, 'expense': {}}
    
    for t in transactions:
        # Monthly data
        month = t.date.strftime('%Y-%m')
        if month not in monthly_data:
            monthly_data[month] = {'income': 0, 'expense': 0}
        monthly_data[month][t.type] += t.amount
        
        # Category data
        if t.category not in category_data[t.type]:
            category_data[t.type][t.category] = 0
        category_data[t.type][t.category] += t.amount
    
    # Convert to format needed for charts
    months = sorted(monthly_data.keys())
    monthly_income = [monthly_data[m]['income'] for m in months]
    monthly_expenses = [monthly_data[m]['expense'] for m in months]
    
    income_categories = list(category_data['income'].keys())
    income_amounts = list(category_data['income'].values())
    
    expense_categories = list(category_data['expense'].keys())
    expense_amounts = list(category_data['expense'].values())
    
    return render_template('analytics.html',
                         months=months,
                         monthly_income=monthly_income,
                         monthly_expenses=monthly_expenses,
                         income_categories=income_categories,
                         income_amounts=income_amounts,
                         expense_categories=expense_categories,
                         expense_amounts=expense_amounts)

@app.route('/cash_flow')
@login_required
def cash_flow():
    user = User.query.get(session['user_id'])
    # Get last 30 days of transactions
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    transactions = Transaction.query.filter(
        Transaction.user_id == user.id,
        Transaction.date >= thirty_days_ago
    ).all()
    
    # Generate cash flow prediction
    daily_balance = {}
    current_balance = 0
    
    for day in range(30):
        date = datetime.utcnow() - timedelta(days=day)
        daily_transactions = [t for t in transactions if t.date.date() == date.date()]
        
        for t in daily_transactions:
            if t.type == 'income':
                current_balance += t.amount
            else:
                current_balance -= t.amount
        
        daily_balance[date.strftime('%Y-%m-%d')] = current_balance
    
    return render_template('cash_flow.html', daily_balance=daily_balance)

@app.route('/financial_education')
@login_required
def financial_education():
    return render_template('financial_education.html')

@app.route('/investor_analysis')
@login_required
def investor_analysis():
    user = User.query.get(session['user_id'])
    
    # Get last 6 months of transactions
    six_months_ago = datetime.now() - timedelta(days=180)
    transactions = Transaction.query.filter(
        Transaction.user_id == user.id,
        Transaction.date >= six_months_ago
    ).order_by(Transaction.date.asc()).all()
    
    # Calculate metrics
    metrics = {
        'income_consistency': round(calculate_income_consistency(transactions)),
        'profit_margin': round(user.net_profit_margin,2),
        'cash_flow': round(user.cash_flow,2),
        'spending_volatility': round(calculate_spending_volatility(transactions),2),
        'savings_rate': round(calculate_savings_rate(transactions),2),
        'credit_utilization': round(user.credit_utilization,2),
        'late_payments': user.late_payments,
        'years_active': user.years_active,
        'industry_growth': get_industry_growth_rating(user.industry)
    }
    
    # Calculate overall investment score
    metrics['investment_score'] = round(calculate_investment_score(user),2)
    
    return render_template('investor_analysis.html', user=user, metrics=metrics)

def calculate_income_consistency(transactions):
    monthly_income = {}
    for t in transactions:
        if t.type == 'income':
            month = t.date.strftime('%Y-%m')
            monthly_income[month] = monthly_income.get(month, 0) + t.amount
    
    if not monthly_income:
        return 0
    
    # Calculate coefficient of variation (lower is better)
    values = list(monthly_income.values())
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    cv = (std_dev / mean) if mean != 0 else float('inf')
    
    # Convert to score (0-100)
    return max(0, min(100, 100 - (cv * 100)))

def calculate_spending_volatility(transactions):
    monthly_expenses = {}
    for t in transactions:
        if t.type == 'expense':
            month = t.date.strftime('%Y-%m')
            monthly_expenses[month] = monthly_expenses.get(month, 0) + t.amount
    
    if not monthly_expenses:
        return 100  # No expenses means perfect score
    
    # Calculate coefficient of variation (lower is better)
    values = list(monthly_expenses.values())
    mean = sum(values) / len(values)
    std_dev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
    cv = (std_dev / mean) if mean != 0 else float('inf')
    
    # Convert to score (0-100)
    return max(0, min(100, 100 - (cv * 100)))

def calculate_savings_rate(transactions):
    total_income = sum(t.amount for t in transactions if t.type == 'income')
    total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
    
    if total_income == 0:
        return 0
    
    savings = total_income - total_expenses
    savings_rate = (savings / total_income) * 100
    return max(0, min(100, savings_rate))

def get_industry_growth_rating(industry):
    # This is a simplified version - in a real app, you'd want to use actual industry data
    growth_ratings = {
        'Technology': 90,
        'Healthcare': 85,
        'E-commerce': 80,
        'Manufacturing': 70,
        'Retail': 65,
        'Agriculture': 60,
        'Other': 50
    }
    return growth_ratings.get(industry, 50)

def calculate_investment_score(business):
    """
    Calculate a comprehensive investment score (0-100) for a business.
    Higher score indicates better investment potential.
    """
    # Get credit score prediction
    business_data = {
        'Annual_Revenue': business.annual_revenue,
        'Net_Profit_Margin': business.net_profit_margin,
        'Total_Assets': business.total_assets,
        'Total_Liabilities': business.total_liabilities,
        'Debt_Equity_Ratio': business.debt_equity_ratio,
        'Cash_Flow': business.cash_flow,
        'Years_Active': business.years_active,
        'Industry': business.industry,
        'Num_Employees': business.num_employees,
        'Credit_History_Length': business.credit_history_length,
        'Late_Payments': business.late_payments,
        'Credit_Utilization': business.credit_utilization,
        'Bankruptcy_Flag': business.bankruptcy_flag
    }
    
    try:
        credit_score = credit_scorer.predict(business_data)
    except Exception as e:
        print("Error predicting credit score:", str(e))
        credit_score = 600  # Default score if prediction fails
    
    # Calculate component scores (0-100)
    financial_score = calculate_financial_score(business)
    growth_score = calculate_growth_score(business)
    stability_score = calculate_stability_score(business)
    credit_score_normalized = round((credit_score - 300) / 5.5, 2)  # Convert to 0-100 scale
    
    # Weight the components
    weights = {
        'financial': 0.35,    # Financial performance
        'growth': 0.25,       # Growth potential
        'stability': 0.20,    # Business stability
        'credit': 0.20        # Credit history
    }
    
    # Calculate weighted score and round to 2 decimal places
    investment_score = round(
        financial_score * weights['financial'] +
        growth_score * weights['growth'] +
        stability_score * weights['stability'] +
        credit_score_normalized * weights['credit'],
        2
    )
    
    # Ensure score stays within 0-100 range
    return max(0, min(100, investment_score))

def calculate_financial_score(business):
    """Calculate financial performance score (0-100)"""
    # Revenue score (up to 30 points)
    revenue_score = round(min(30, business.annual_revenue / 1000000), 2)
    
    # Profit margin score (up to 30 points)
    margin_score = round(min(30, business.net_profit_margin * 100), 2)
    
    # Cash flow score (up to 20 points)
    cash_flow_score = round(min(20, business.cash_flow / 50000), 2)
    
    # Debt ratio score (up to 20 points)
    debt_score = round(max(0, 20 - (business.debt_equity_ratio * 10)), 2)
    
    return round(revenue_score + margin_score + cash_flow_score + debt_score, 2)

def calculate_growth_score(business):
    """Calculate growth potential score (0-100)"""
    # Industry growth potential (up to 40 points)
    industry_score = get_industry_growth_potential(business.industry)
    
    # Revenue growth trend (up to 30 points)
    revenue_growth = round(min(30, business.annual_revenue / 2000000), 2)
    
    # Employee growth potential (up to 30 points)
    employee_growth = round(min(30, business.num_employees / 10), 2)
    
    return round(industry_score + revenue_growth + employee_growth, 2)

def calculate_stability_score(business):
    """Calculate business stability score (0-100)"""
    # Years in business (up to 30 points)
    years_score = round(min(30, business.years_active * 3), 2)
    
    # Credit history length (up to 25 points)
    credit_history_score = round(min(25, business.credit_history_length * 2.5), 2)
    
    # Late payments penalty (up to 20 points deduction)
    late_payments_penalty = round(min(20, business.late_payments * 4), 2)
    
    # Bankruptcy penalty (25 points deduction)
    bankruptcy_penalty = 25 if business.bankruptcy_flag else 0
    
    # Credit utilization penalty (up to 25 points deduction)
    utilization_penalty = round(min(25, business.credit_utilization * 25), 2)
    
    return round(max(0, years_score + credit_history_score - late_payments_penalty - bankruptcy_penalty - utilization_penalty), 2)

def get_industry_growth_potential(industry):
    """Get industry-specific growth potential score"""
    industry_growth_rates = {
        'Technology': 90,
        'Healthcare': 85,
        'E-commerce': 80,
        'Manufacturing': 70,
        'Retail': 65,
        'Agriculture': 60,
        'Other': 50
    }
    return industry_growth_rates.get(industry, 50)

@app.route('/profile')
@login_required
def profile():
    user = User.query.get(session['user_id'])
    return render_template('profile.html', user=user)

@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])
    
    # Update basic information
    if 'business_id' in request.form:
        user.business_id = request.form['business_id']
        user.business_type = request.form['business_type']
        user.industry = request.form['industry']
        user.years_active = int(request.form['years_active'])
        user.num_employees = int(request.form['num_employees'])
    
    # Update financial information
    if 'annual_revenue' in request.form:
        user.annual_revenue = float(request.form['annual_revenue'])
        user.net_profit_margin = float(request.form['net_profit_margin'])
        user.total_assets = float(request.form['total_assets'])
        user.total_liabilities = float(request.form['total_liabilities'])
        user.cash_flow = float(request.form['cash_flow'])
        
        # Calculate debt-equity ratio
        if user.total_liabilities > 0:
            user.debt_equity_ratio = user.total_liabilities / (user.total_assets - user.total_liabilities)
        else:
            user.debt_equity_ratio = 0.0
    
    # Update credit information
    if 'credit_history_length' in request.form:
        user.credit_history_length = int(request.form['credit_history_length'])
        user.late_payments = int(request.form['late_payments'])
        user.credit_utilization = float(request.form['credit_utilization'])
        user.bankruptcy_flag = 'bankruptcy_flag' in request.form
    
    # Update account information
    if 'username' in request.form:
        new_username = request.form['username']
        if new_username != user.username and User.query.filter_by(username=new_username).first():
            flash('Username already exists')
            return redirect(url_for('profile'))
        user.username = new_username
        
        new_email = request.form['email']
        if new_email != user.email and User.query.filter_by(email=new_email).first():
            flash('Email already exists')
            return redirect(url_for('profile'))
        user.email = new_email
        
        if request.form['password']:
            user.set_password(request.form['password'])
    
    db.session.commit()
    flash('Profile updated successfully!')
    return redirect(url_for('profile'))

def create_sample_transactions(user_id):
    # Sample income categories and amounts
    income_categories = {
        'Sales': {'min': 5000, 'max': 15000, 'frequency': 0.4},
        'Services': {'min': 3000, 'max': 8000, 'frequency': 0.3},
        'Investments': {'min': 1000, 'max': 5000, 'frequency': 0.2},
        'Other': {'min': 500, 'max': 2000, 'frequency': 0.1}
    }
    
    # Sample expense categories and amounts
    expense_categories = {
        'Rent': {'min': 2000, 'max': 4000, 'frequency': 0.2},
        'Salaries': {'min': 5000, 'max': 10000, 'frequency': 0.3},
        'Marketing': {'min': 1000, 'max': 3000, 'frequency': 0.2},
        'Utilities': {'min': 500, 'max': 1500, 'frequency': 0.1},
        'Supplies': {'min': 300, 'max': 1000, 'frequency': 0.1},
        'Other': {'min': 200, 'max': 800, 'frequency': 0.1}
    }
    
    # Create transactions for the last 6 months
    today = datetime.now()
    for i in range(180):  # 6 months of daily transactions
        date = today - timedelta(days=i)
        
        # Add income transactions
        for category, details in income_categories.items():
            if random.random() < details['frequency']:
                amount = random.randint(details['min'], details['max'])
                transaction = Transaction(
                    amount=amount,
                    type='income',
                    category=category,
                    description=f"Monthly {category}",
                    date=date,
                    user_id=user_id
                )
                db.session.add(transaction)
        
        # Add expense transactions
        for category, details in expense_categories.items():
            if random.random() < details['frequency']:
                amount = random.randint(details['min'], details['max'])
                transaction = Transaction(
                    amount=amount,
                    type='expense',
                    category=category,
                    description=f"Monthly {category}",
                    date=date,
                    user_id=user_id
                )
                db.session.add(transaction)
    
    db.session.commit()

@app.route('/add_sample_data')
@login_required
def add_sample_data():
    user = User.query.get(session['user_id'])
    
    # Delete existing transactions
    Transaction.query.filter_by(user_id=user.id).delete()
    
    # Create new sample transactions
    create_sample_transactions(user.id)
    
    flash('Sample transaction data has been added to your account!')
    return redirect(url_for('analytics'))

@app.route('/register_investor', methods=['GET', 'POST'])
def register_investor():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        investment_preference = request.form['investment_preference']
        investment_amount = float(request.form['investment_amount'])

        if password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register_investor'))

        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register_investor'))

        if User.query.filter_by(email=email).first():
            flash('Email already exists!', 'danger')
            return redirect(url_for('register_investor'))

        user = User(
            username=username,
            email=email,
            is_investor=True,
            investment_preference=investment_preference,
            max_investment_amount=investment_amount
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('investor_register.html')

@app.route('/investor_dashboard')
@login_required
def investor_dashboard():
    if not current_user.is_investor:
        flash('Access denied. This page is for investors only.', 'danger')
        return redirect(url_for('dashboard'))

    # Get investor's investments
    investments = Investment.query.filter_by(investor_id=current_user.id).all()
    
    # Calculate dashboard metrics
    total_investments = sum(investment.amount for investment in investments)
    active_investments = sum(1 for investment in investments if investment.status == 'active')
    avg_return = sum(investment.return_rate for investment in investments) / len(investments) if investments else 0
    risk_score = sum(investment.risk_score for investment in investments) / len(investments) if investments else 0

    # Get available businesses
    available_businesses = User.query.filter_by(is_investor=False).all()

    return render_template('investor_dashboard.html',
                         investments=investments,
                         total_investments=total_investments,
                         active_investments=active_investments,
                         avg_return=avg_return,
                         risk_score=risk_score,
                         available_businesses=available_businesses)

@app.route('/business_profile/<int:business_id>')
@login_required
def business_profile(business_id):
    business = User.query.get_or_404(business_id)
    if business.is_investor:
        flash('Invalid business profile.', 'danger')
        return redirect(url_for('investor_dashboard'))

    # Generate sample data for charts
    revenue_data = [random.randint(5000, 15000) for _ in range(6)]
    expenses_data = [random.randint(3000, 8000) for _ in range(6)]
    growth_data = [random.randint(5, 20) for _ in range(6)]

    return render_template('business_profile.html',
                         business=business,
                         revenue_data=revenue_data,
                         expenses_data=expenses_data,
                         growth_data=growth_data)

@app.route('/invest/<int:business_id>', methods=['GET', 'POST'])
@login_required
def invest(business_id):
    if not current_user.is_investor:
        flash('Access denied. This page is for investors only.', 'danger')
        return redirect(url_for('dashboard'))

    business = User.query.get_or_404(business_id)
    if business.is_investor:
        flash('Invalid business.', 'danger')
        return redirect(url_for('investor_dashboard'))

    if request.method == 'POST':
        amount = float(request.form['amount'])
        
        if amount > current_user.max_investment_amount:
            flash('Investment amount exceeds your maximum limit.', 'danger')
            return redirect(url_for('invest', business_id=business_id))

        # Calculate risk score and return rate based on business metrics
        risk_score = calculate_risk_score(business)
        return_rate = calculate_return_rate(business, amount)

        investment = Investment(
            investor_id=current_user.id,
            business_id=business_id,
            amount=amount,
            risk_score=risk_score,
            return_rate=return_rate
        )

        db.session.add(investment)
        db.session.commit()

        flash('Investment successful!', 'success')
        return redirect(url_for('investor_dashboard'))

    return render_template('invest.html', business=business)

def calculate_risk_score(business):
    """
    Calculate a comprehensive risk score (0-100) for a business investment.
    Lower score indicates lower risk.
    """
    # Get credit score prediction
    business_data = {
        'Annual_Revenue': business.annual_revenue,
        'Net_Profit_Margin': business.net_profit_margin,
        'Total_Assets': business.total_assets,
        'Total_Liabilities': business.total_liabilities,
        'Debt_Equity_Ratio': business.debt_equity_ratio,
        'Cash_Flow': business.cash_flow,
        'Years_Active': business.years_active,
        'Industry': business.industry,
        'Num_Employees': business.num_employees,
        'Credit_History_Length': business.credit_history_length,
        'Late_Payments': business.late_payments,
        'Credit_Utilization': business.credit_utilization,
        'Bankruptcy_Flag': business.bankruptcy_flag
    }
    
    try:
        credit_score = credit_scorer.predict(business_data)
    except Exception as e:
        print("Error predicting credit score:", str(e))
        credit_score = 600  # Default score if prediction fails
    
    # Convert credit score to risk score (inverse relationship)
    # Credit scores range from 300-850, we want risk scores from 0-100
    risk_score = 100 - ((credit_score - 300) / 5.5)
    
    # Adjust risk score based on additional factors
    adjustments = {
        'years_active': min(10, business.years_active) * 2,  # Up to 20 points for experience
        'bankruptcy': 20 if business.bankruptcy_flag else 0,  # 20 points penalty for bankruptcy
        'late_payments': min(15, business.late_payments * 3),  # Up to 15 points for late payments
        'credit_utilization': min(10, business.credit_utilization * 10),  # Up to 10 points for high utilization
        'debt_ratio': min(15, business.debt_equity_ratio * 10)  # Up to 15 points for high debt
    }
    
    # Apply adjustments to risk score
    risk_score += sum(adjustments.values())
    
    # Ensure risk score stays within 0-100 range
    return max(0, min(100, risk_score))

def calculate_return_rate(business, investment_amount):
    """
    Calculate expected return rate based on business metrics and investment amount.
    Returns annual percentage rate (APR).
    """
    # Base return components
    financial_return = calculate_financial_return(business)
    growth_return = calculate_growth_return(business)
    risk_adjustment = calculate_risk_adjustment(business)
    
    # Investment size adjustment (larger investments get better rates)
    size_adjustment = round(min(2.0, investment_amount / 100000), 2)  # Cap at 2% for very large investments
    
    # Calculate base return rate
    base_rate = round(
        financial_return * 0.5 +  # Financial performance is most important
        growth_return * 0.3 +     # Growth potential second
        risk_adjustment * 0.2,    # Risk profile third
        2
    )
    
    # Apply investment size adjustment
    final_rate = round(base_rate + size_adjustment, 2)
    
    # Ensure return rate is within reasonable bounds
    return max(2.0, min(25.0, final_rate))

def calculate_financial_return(business):
    """Calculate return component based on financial performance"""
    # Profit margin contribution
    margin_contribution = round(business.net_profit_margin * 0.5, 2)
    
    # Revenue growth contribution
    growth_contribution = round(min(10.0, business.annual_revenue / 1000000), 2)
    
    # Cash flow contribution
    cash_flow_contribution = round(min(5.0, business.cash_flow / 10000), 2)
    
    return round(margin_contribution + growth_contribution + cash_flow_contribution, 2)

def calculate_growth_return(business):
    """Calculate return component based on growth potential"""
    # Industry growth potential
    industry_growth = round(get_industry_growth_potential(business.industry) / 10, 2)
    
    # Business size growth potential
    size_growth = round(min(5.0, business.num_employees / 20), 2)
    
    # Revenue growth trend
    revenue_growth = round(min(5.0, business.annual_revenue / 2000000), 2)
    
    return round(industry_growth + size_growth + revenue_growth, 2)

def calculate_risk_adjustment(business):
    """Calculate risk-based return adjustment"""
    risk_score = calculate_risk_score(business)
    
    # Higher risk should yield higher potential returns
    if risk_score < 30:
        return 2.0  # Low risk premium
    elif risk_score < 60:
        return 4.0  # Medium risk premium
    else:
        return 6.0  # High risk premium

def calculate_cash_flow(transactions):
    """
    Calculate cash flow based on recent transactions.
    Returns annual cash flow in dollars.
    """
    if not transactions:
        return 0.0
    
    # Get transactions from the last 6 months for better accuracy
    six_months_ago = datetime.utcnow() - timedelta(days=180)
    recent_transactions = [t for t in transactions if t.date >= six_months_ago]
    
    if not recent_transactions:
        return 0.0
    
    # Calculate monthly cash flow
    monthly_cash_flow = {}
    for t in recent_transactions:
        month = t.date.strftime('%Y-%m')
        if month not in monthly_cash_flow:
            monthly_cash_flow[month] = {'income': 0.0, 'expense': 0.0}
        
        if t.type == 'income':
            monthly_cash_flow[month]['income'] += t.amount
        else:
            monthly_cash_flow[month]['expense'] += t.amount
    
    # Calculate net cash flow for each month
    monthly_net_cash_flow = {
        month: data['income'] - data['expense']
        for month, data in monthly_cash_flow.items()
    }
    
    # Calculate average monthly cash flow
    avg_monthly_cash_flow = sum(monthly_net_cash_flow.values()) / len(monthly_net_cash_flow)
    
    # Annualize the cash flow and round to 2 decimal places
    annual_cash_flow = round(avg_monthly_cash_flow * 12, 2)
    
    return annual_cash_flow

def calculate_cash_flow_metrics(transactions):
    """
    Calculate detailed cash flow metrics for analysis.
    Returns a dictionary with various cash flow metrics.
    """
    if not transactions:
        return {
            'annual_cash_flow': 0.0,
            'monthly_avg_income': 0.0,
            'monthly_avg_expense': 0.0,
            'cash_flow_consistency': 0.0,
            'operating_cash_flow': 0.0
        }
    
    # Get transactions from the last 6 months
    six_months_ago = datetime.utcnow() - timedelta(days=180)
    recent_transactions = [t for t in transactions if t.date >= six_months_ago]
    
    if not recent_transactions:
        return {
            'annual_cash_flow': 0.0,
            'monthly_avg_income': 0.0,
            'monthly_avg_expense': 0.0,
            'cash_flow_consistency': 0.0,
            'operating_cash_flow': 0.0
        }
    
    # Calculate monthly metrics
    monthly_metrics = {}
    for t in recent_transactions:
        month = t.date.strftime('%Y-%m')
        if month not in monthly_metrics:
            monthly_metrics[month] = {
                'income': 0.0,
                'expense': 0.0,
                'operating_income': 0.0,
                'operating_expense': 0.0
            }
        
        if t.type == 'income':
            monthly_metrics[month]['income'] += t.amount
            if t.category in ['Sales', 'Services']:  # Operating income
                monthly_metrics[month]['operating_income'] += t.amount
        else:
            monthly_metrics[month]['expense'] += t.amount
            if t.category in ['Rent', 'Salaries', 'Utilities', 'Supplies']:  # Operating expenses
                monthly_metrics[month]['operating_expense'] += t.amount
    
    # Calculate averages
    num_months = len(monthly_metrics)
    monthly_avg_income = sum(m['income'] for m in monthly_metrics.values()) / num_months
    monthly_avg_expense = sum(m['expense'] for m in monthly_metrics.values()) / num_months
    monthly_avg_operating_income = sum(m['operating_income'] for m in monthly_metrics.values()) / num_months
    monthly_avg_operating_expense = sum(m['operating_expense'] for m in monthly_metrics.values()) / num_months
    
    # Calculate cash flow consistency (coefficient of variation)
    monthly_net_cash_flows = [m['income'] - m['expense'] for m in monthly_metrics.values()]
    mean_cash_flow = sum(monthly_net_cash_flows) / num_months
    if mean_cash_flow != 0:
        std_dev = (sum((x - mean_cash_flow) ** 2 for x in monthly_net_cash_flows) / num_months) ** 0.5
        cash_flow_consistency = (1 - (std_dev / abs(mean_cash_flow))) * 100
    else:
        cash_flow_consistency = 0.0
    
    # Calculate annual metrics
    annual_cash_flow = round(monthly_avg_income * 12 - monthly_avg_expense * 12, 2)
    operating_cash_flow = round((monthly_avg_operating_income - monthly_avg_operating_expense) * 12, 2)
    
    return {
        'annual_cash_flow': annual_cash_flow,
        'monthly_avg_income': round(monthly_avg_income, 2),
        'monthly_avg_expense': round(monthly_avg_expense, 2),
        'cash_flow_consistency': round(cash_flow_consistency, 2),
        'operating_cash_flow': operating_cash_flow
    }

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True) 