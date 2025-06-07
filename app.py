#Importing necessary libraries 
import nltk
import numpy as np
import pandas as pd
from flask import *
import mysql.connector
from pytube import YouTube
# import spacy
import random 
import youtube_dl
# from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from gensim.summarization.summarizer import summarize
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from nltk.tokenize import sent_tokenize
import os
import cv2
import subprocess
from pytube import YouTube
from urllib.parse import quote as url_quote
from contextvars import ContextVar
from googletrans import Translator
nltk.download('punkt_tab')
from tensorflow.keras.models import load_model

db=mysql.connector.connect(user="root",password="",port='3306',database='video')
cur=db.cursor()

app = Flask(__name__)
app.secret_key = "##################################"
app.config['UPLOAD_FOLDER'] = 'videos'



@app.route('/')
def index():

    return render_template('index.html')


@app.route('/about')
def about():

    return render_template('about.html')


@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')


@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                msg = f'Account created for {username}!'
                flash("Registered successfully","success")
                return render_template("login.html",msg=msg)
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')




@app.route('/userhome')
def userhome():

    return render_template('userhome.html')

def load_lstm_model():
    print("Loading LSTM model for summarization.")
    model = load_model("model.h5")  
    return model

model = load_lstm_model()


def get_summary(text, method, percentage):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return "Transcript is too short to summarize. Original Transcript: " + text
    
    try:
        if method == 'extractive':
            print("Using LSTM model for extractive summarization")
            return summarize(text, ratio=float(percentage)/100)
            word_count = len(summary.split())  # Calculate word count for extractive
            return summary, word_count  # Ensure you have the correct function for summarizing
        elif method == 'abstractive':
            summarizer = pipeline("summarization")  # Ensure you have the correct pipeline for summarization
            summary = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
            word_count = len(summary.split())
            return summary,word_count
    except ValueError as ve:
        return f"Error during summarization: {ve}. Original Transcript: {text}"

import re

def extract_video_id(url):
    # Regular expression to match YouTube video IDs
    match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None

from googletrans import Translator
def translate_summary(text, language):
    """Translate the summary text to the specified language."""
    translator = Translator()
    trans_summary = translator.translate(text, dest=language)
    translated_text = trans_summary.text  # Get translated text
    pronunciation = trans_summary.pronunciation if trans_summary.pronunciation else "No pronunciation available"
    return translated_text, pronunciation



@app.route('/summary', methods=['GET', 'POST'])
def summary():
    word_count = 0
    if request.method == 'POST':
        url = request.form['url']
        method = request.form['method']
        percentage = request.form['percentage']
        language = request.form['language']

        match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        video_id = match.group(1) if match else None
        
        transcript = get_transcript(video_id)
        summary = get_summary(transcript, method, percentage)

        # Get summary only if transcript exists
        if transcript:
            summary = get_summary(transcript, method, percentage)
            words = summary.split(" ")
            word_count = 0
            for word in words:
                word_count += 1
        else:
            summary = "Transcript not available."

        # Only translate if summary is available and valid
        if summary and summary != "Transcript not available.":
            translated_text, pronunciation = translate_summary(summary, language)

        words = translated_text.split(" ")
        translted_summary_word_count = 0
        for word in words:
            translted_summary_word_count += 1

       

        return render_template('summary.html', 
                                summary=summary, 
                                transcript=transcript, 
                                trans_summary=translated_text, 
                                pronunciation=pronunciation,
                                word_count=word_count,
                                translted_summary_word_count = translted_summary_word_count
                                ) #

    return render_template('summary.html', summary=None)





def get_transcript(video_id):
    # Extract transcript using YouTubeTranscriptApi
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    transcript = ' '.join([entry['text'] for entry in transcript_list])
    
    # Split transcript into words
    words = transcript.split()
    
    # Add a full stop after every 15-20 words
    for i in range(15, len(words), 15 + random.randint(0, 5)):
        words[i] += '.'
    
    # Join words back into sentences
    updated_transcript = ' '.join(words)
    
    return updated_transcript

def get_summary(text, method, percentage):
    sentences = sent_tokenize(text)
    if len(sentences) <= 1:
        return "Transcript is too short to summarize. Original Transcript: " + text
    
    try:
        if method == 'extractive':
            return summarize(text, ratio=float(percentage)/100)  # Ensure you have the correct function for summarizing
        elif method == 'abstractive':
            summarizer = pipeline("summarization")  # Ensure you have the correct pipeline for summarization
            summary = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']
            return summary
    except ValueError as ve:
        return f"Error during summarization: {ve}. Original Transcript: {text}"





@app.route('/video',methods=['POST','GET'])
def video():
    return render_template('video.html')


@app.route('/summary2', methods=['POST','GET'])
def summary2():
    word_count = 0
    if request.method == 'POST':
        url = request.form['url']
        method = request.form['method']
        percentage = request.form['percentage']
        # language = request.form['language']

        match = re.search(r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})', url)
        video_id = match.group(1) if match else None
        
        transcript = get_transcript(video_id)
        summary = get_summary(transcript, method, percentage)

        # Get summary only if transcript exists
        # if transcript:
        #     summary = get_summary(transcript, method, percentage)
        #     words = summary.split(" ")
        #     word_count = 0
        #     for word in words:
        #         word_count += 1
        # else:
        #     summary = "Transcript not available."

        # # Only translate if summary is available and valid
        # if summary and summary != "Transcript not available.":
        #     translated_text, pronunciation = translate_summary(summary)

        # words = translated_text.split(" ")
        # translted_summary_word_count = 0
        # for word in words:
        #     translted_summary_word_count += 1

       

        return render_template('summary2.html', 
                                summary=summary, 
                                transcript=transcript, 
                                # trans_summary=translated_text, 
                                # pronunciation=pronunciation,
                                # word_count=word_count,
                                # translted_summary_word_count = translted_summary_word_count
                                ) #
    return render_template('summary2.html')






UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

@app.route('/play', methods=['GET', 'POST'])
def play():
    if request.method == 'POST':
        file = request.files['file']
        start_time = request.form['start_time']
        end_time = request.form['end_time']
        start_ms = timestamp_to_ms(start_time)
        end_ms = timestamp_to_ms(end_time)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        output_file = process_video(filepath, start_ms, end_ms)
        video_url = url_for('static', filename=f'outputs/{output_file}')
        msg = "Video has processed successfully"
        path = '/static/outputs/output_video.mp4'
        return render_template('play.html', video_url=video_url, path =path, msg=msg)
    return render_template('play.html', video_url=None)

def timestamp_to_ms(timestamp):
    h, m, s = map(int, timestamp.split(':'))
    return (h * 3600 + m * 60 + s) * 1000

def process_video(input_path, start_time, end_time):
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    output_file = 'output_video.mp4'
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_file)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), 20, (frame_width, frame_height))
    cap.set(cv2.CAP_PROP_POS_MSEC, start_time)

    while cap.get(cv2.CAP_PROP_POS_MSEC) < end_time:
        ret, frame = cap.read()
        if ret == True:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    return output_file


@app.route('/download', methods=['POST','GET'])
def download():
    if request.method == 'POST':
        link = request.form.get('link')
        start_time = request.form.get('start_time')
        end_time = request.form.get('end_time')
        video_id = link.split('v=')[-1].split('&')[0]  # Extract video ID from the YouTube URL
        return render_template('video.html', video_id=video_id, start_time=start_time, end_time=end_time)
    return render_template('video.html')





if __name__ =='__main__':
    app.run(debug=True)


    