# Chatbot
A simple chatbot in python using sequence to sequence model

1. <p><b>Run the requirements.txt for 64-bit python installation.</b> <br/> <code>py -m pip install -r requirements.txt</code><br/><br/></p>

2. <p><b>Put the subtitle files inside a folder or multiple folders and put these folders inside the subs folder. </b> <br/>So the idea is, you can put a single season or the whole series' subtitles in a folder and you can have multiple such type of folders. Then you have to just put the folders under subs folder.<br/><br/></p>

3. <p><b>Run create_dataset.py </b><br/>This creates 2 .npy files containing questions and answers. <br/> <code>py create_dataset.py</code><br/><br/></p>

4. <p><b>Run chatbot.py to train the bot using the above created dataset.</b><br/><code>py chatbot.py</code><br/><br/></p>

5. <p><b>To chat with the bot, run inference.py.</b><br/><code>py inference.py</code><br/><br/></p>
