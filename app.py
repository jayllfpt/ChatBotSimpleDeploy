from flask import Flask, request, render_template
from flask_cors import CORS
from bots.bot import *

app = Flask(__name__)
CORS(app)
# Loading our model and tokenizer into our backend. Now our backend can use these to make prediction.
data = load_data()
model = load_model()
data_embeddings = embeddings_data(data, model)

# When you go to localhost:5000 using a web browswer, it will render index.html's content
@app.route('/') 
def index(): 
    return render_template('index.html')

# The Frontend will make API call to this localhost:5000/predict?text=TheMovieIsTerrible, and backend will return "The review is negative"
@app.route('/predict', methods=['GET'])
def predict():
    text = request.args.get('text', '') # Todo: Add urlib to parse the encoded text https://www.urldecoder.io/python/ 
    _bot_reply = getAnswer(text, data, data_embeddings, model)
    print(_bot_reply)
    if _bot_reply['Score'] < 0.8:
        return default_message
    else:
        # bot_reply = "\nAnswer:" + _bot_reply['Answer']
        # bot_reply += "\nSimiarity Question:" + _bot_reply['Simiarity Q'] 
        # bot_reply += "\nScore:" + str(_bot_reply['Score'])  
        return _bot_reply['Answer']