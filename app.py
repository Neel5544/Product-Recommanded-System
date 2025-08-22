from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'Neel chauhan'

def init_db():
    conn = sqlite3.connect('users.db', check_same_thread=False, timeout=10)
    c = conn.cursor()
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY, 
                    username TEXT UNIQUE, 
                    password TEXT)''')
    conn.commit()
    conn.close()

init_db()

def load_data():
    try:
        df = pd.read_csv("amazon_cleaned.csv")
        df = df.dropna(subset=['product_id', 'product_name', 'discounted_price', 'actual_price', 'rating', 'img_link'])
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df = df.dropna(subset=['rating'])
        return df
    except Exception as e:
        print("Error loading dataset:", e)
        return pd.DataFrame()

data = load_data()

def compute_similarity():
    if data.empty:
        return None

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    text_features = data['product_name'] + " " + data.get('about_product', '')
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_features.fillna(""))
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

similarity_matrix = compute_similarity()

@app.route('/')
def home():
    if 'username' not in session:
        flash("You must log in first!", "warning")
        return redirect(url_for('login'))

    if data.empty:
        flash("Product data is missing. Upload `amazon_cleaned.csv`!", "danger")
        return "No product data found", 500

    top_rated = data.sort_values(by='rating', ascending=False).head(52).to_dict(orient='records')
    return render_template('index.html', username=session['username'], top_rated_products=top_rated)

def recommend_products(product_id, num_recommendations=20):
    if similarity_matrix is None:
        return []

    product_indices = data.index[data['product_id'] == product_id].tolist()
    
    if not product_indices:
        return []
    
    idx = product_indices[0]
    sim_scores = sorted(enumerate(similarity_matrix[idx]), key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    recommended_indices = [i[0] for i in sim_scores]
    return data.iloc[recommended_indices].to_dict(orient='records')

@app.route('/product/<product_id>')
def product_page(product_id):
    if 'username' not in session:
        flash("You must log in first!", "warning")
        return redirect(url_for('login'))
    
    product = data[data['product_id'] == product_id].to_dict(orient='records')
    if not product:
        return "Product not found", 404

    recommended = recommend_products(product_id)
    return render_template('product.html', product=product[0], recommended_products=recommended)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT password FROM users WHERE username=?", (username,))
        result = c.fetchone()
        conn.close()

        if result and check_password_hash(result[0], password):
            session['username'] = username
            flash("Login successful!", "success")
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password!", "danger")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect('users.db')
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
            conn.commit()
            conn.close()
            flash("Signup successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("Logged out successfully!", "success")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
