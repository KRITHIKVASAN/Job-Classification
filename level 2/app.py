from flask import Flask, render_template, request, redirect, url_for, session
from flask_wtf.csrf import CSRFProtect
from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField
from wtforms.validators import DataRequired
from gensim.models import FastText
import json
import pickle
import numpy as np
from utils import docvecs

app = Flask(__name__)
app.config['SECRET_KEY'] = 'bdc5d93a9ae3322b3b857ddf88ba2565'
csrf = CSRFProtect(app)

class JobListingForm(FlaskForm):
    title = StringField('Job Title', validators=[DataRequired()])
    description = TextAreaField('Job Description', validators=[DataRequired()])
    company = StringField('Company Name', validators=[DataRequired()])
    category = StringField('Category')

# Load job data from JSON file
job_data = []
with open("job_data.json", "r") as f:
    job_data = json.load(f)

# Create a function to group jobs by category
def group_jobs_by_category(jobs):
    categories = {}
    for job in jobs:
        category = job['job_category']
        if category not in categories:
            categories[category] = []
        categories[category].append(job)
    return categories


@app.route('/')
def index():
    jobs_by_category = group_jobs_by_category(job_data)
    form = JobListingForm()  # Create an instance of the form
    return render_template('index.html', jobs_by_category=jobs_by_category, form=form)

@app.route('/category/<category>')
def category(category):
    category_jobs = [job for job in job_data if job['job_category'] == category]
    return render_template('category.html', category=category, jobs=category_jobs)

@app.route('/job/<int:index>')
def job_details(index):
    job = job_data[index]
    return render_template('job.html', job=job)

@app.route('/create_job_listing', methods=['GET', 'POST'])
def create_job_listing():
    print("hi")
    form = JobListingForm()
    classification_result = None  # Initialize classification_result

    if request.method == 'POST' and 'listing' in request.form.get('button', ''):
        new_job_listing = {
            "job_title": form.title.data,
            "company": form.company.data,
            "job_desc": form.description.data,
            "job_category": form.category.data
        }

        # Update the custom index based on the current length of job_data
        new_job_listing['custom_index'] = len(job_data)

        job_data.append(new_job_listing)

        # Save the updated job_data to the JSON file
        with open("job_data.json", "w") as f:
            json.dump(job_data, f, indent=4)

        return redirect(url_for('index'))

    if request.method == 'POST' and 'Classify' in request.form.get('button', ''):
        # Perform classification when the "Classify" button is clicked
        description = form.description.data
        print(description)
        tokenized_description = description.split(' ')

        descFT_model = FastText.load("desc_FT.model")
        descFT_wv = descFT_model.wv

        # Generate vector representation of the tokenized description
        bbcFT_dvs = docvecs(descFT_wv, [tokenized_description])

        with open("descFT_LR.pkl", 'rb') as lr_model_file:
            lr_model = pickle.load(lr_model_file)

        # Predict the label
        y_pred = lr_model.predict(bbcFT_dvs)
        classification_result = y_pred[0]

    return render_template('create_job_listing.html', form=form, classification_result=classification_result)

@app.route('/search', methods=['GET', 'POST'])
def search():
    search_query = request.form.get('search_query', '')
    results = perform_search(search_query)
    return render_template('search_results.html', search_query=search_query, results=results)

def perform_search(search_query):
    # Basic search logic: search in job titles
    search_results = [job for job in job_data if search_query.lower() in job['job_title'].lower()]
    return search_results

if __name__ == '__main__':
    app.run(debug=True)
