import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    confidence = model.decision_function(vec)[0]
    prob = 1 / (1 + np.exp(-confidence))  # sigmoid to get 0-1 score
    label = "Real News" if pred == 1 else "Fake News"
    return label, prob

def get_text_stats(text):
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    reading_time_min = word_count / 200  # approx 200 WPM
    return word_count, char_count, reading_time_min

 

# --- STREAMLIT APP ---

st.set_page_config(page_title="📰 Fake News Classifier", page_icon="📰", layout="wide")

with st.sidebar:
    st.title("About")
    st.markdown("""
    This app uses a **Passive Aggressive Classifier** to detect whether news is **Real** or **Fake**.
    
    - Trained on a popular Fake/Real news dataset.
    - Uses TF-IDF features and classic ML (no deep learning).
    - Accuracy: 0.9949
    
    ### Usage:
    1. Paste news text or headline.
    2. Click *Classify*.
    3. See the prediction and confidence score.
    """)
    st.markdown("---")
    st.markdown("### Sample news to try :")
    sample_news = st.selectbox(
        "Choose an example:",
        options=[
           "WEST PALM BEACH, Fla./WASHINGTON (Reuters) - The White House said on Friday it was set to kick off talks next week with Republican and Democratic congressional leaders on immigration policy, government spending and other issues that need to be wrapped up early in the new year. The expected flurry of legislative activity comes as Republicans and Democrats begin to set the stage for midterm congressional elections in November. President Donald Trumpâ€™s Republican Party is eager to maintain control of Congress while Democrats look for openings to wrest seats away in the Senate and the House of Representatives. On Wednesday, Trumpâ€™s budget chief Mick Mulvaney and legislative affairs director Marc Short will meet with Senate Majority Leader Mitch McConnell and House Speaker Paul Ryan - both Republicans - and their Democratic counterparts, Senator Chuck Schumer and Representative Nancy Pelosi, the White House said. That will be followed up with a weekend of strategy sessions for Trump, McConnell and Ryan on Jan. 6 and 7 at the Camp David presidential retreat in Maryland, according to the White House. The Senate returns to work on Jan. 3 and the House on Jan. 8. Congress passed a short-term government funding bill last week before taking its Christmas break, but needs to come to an agreement on defense spending and various domestic programs by Jan. 19, or the government will shut down. Also on the agenda for lawmakers is disaster aid for people hit by hurricanes in Puerto Rico, Texas and Florida, and by wildfires in California. The House passed an $81 billion package in December, which the Senate did not take up. The White House has asked for a smaller figure, $44 billion. Deadlines also loom for soon-to-expire protections for young adult immigrants who entered the country illegally as children, known as â€œDreamers.â€ In September, Trump ended Democratic former President Barack Obamaâ€™s Deferred Action for Childhood Arrivals (DACA) program, which protected Dreamers from deportation and provided work permits, effective in March, giving Congress until then to devise a long-term solution. Democrats, some Republicans and a number of large companies have pushed for DACA protections to continue. Trump and other Republicans have said that will not happen without Congress approving broader immigration policy changes and tougher border security. Democrats oppose funding for a wall promised by Trump along the U.S.-Mexican border.  â€œThe Democrats have been told, and fully understand, that there can be no DACA without the desperately needed WALL at the Southern Border and an END to the horrible Chain Migration & ridiculous Lottery System of Immigration etc,â€ Trump said in a Twitter post on Friday. Trump wants to overhaul immigration rules for extended families and others seeking to live in the United States. Republican U.S. Senator Jeff Flake, a frequent critic of the president, said he would work with Trump to protect Dreamers. â€œWe can fix DACA in a way that beefs up border security, stops chain migration for the DREAMers, and addresses the unfairness of the diversity lottery. If POTUS (Trump) wants to protect these kids, we want to help him keep that promise,â€ Flake wrote on Twitter. Congress in early 2018 also must raise the U.S. debt ceiling to avoid a government default. The U.S. Treasury would exhaust all of its borrowing options and run dry of cash to pay its bills by late March or early April if Congress does not raise the debt ceiling before then, according to the nonpartisan Congressional Budget Office. Trump, who won his first major legislative victory with the passage of a major tax overhaul this month, has also promised a major infrastructure plan. ",
            "spokeswoman said. Seventeen Vietnamese sailors were taken captive in the south of the Philippines when pro-Islamic State militants attacked commercial ships in the Sulu and Celebes seas, generating millions of dollars from ransom payments. Four Vietnamese sailors were found on a remote island in Tawi-tawi, but one was already dead from illness, Captain Jo-Ann Petinglay told reporters. She said Philippines marines were pursuing a small group of militants who had fled and abandoned the captives. Of the 17 Vietnamese sailors abducted in February, 10 were rescued in earlier operations and two were killed by the militants while attempting to escape, she said. The rest were brought to nearby Jolo island, she added without elaborating.  The military said the militants, known for kidnapping, bombing and beheading captives, are still holding 16 captives on Jolo island, including seven foreigners, including a Dutch and a Japanese national.",
            "A centerpiece of Donald Trump s campaign, and now his presidency, has been his white supremacist ",
            "JAKARTA/TIMIKA, Indonesia (Reuters) - Armed separatists have occupied five villages in Indonesia s ",
            "It s gotta be hard these days to be John McCain. Standing up to Donald Trump is pretty unpopular among Republicans these days, and we ve seen the proof in how quickly the base turned on some otherwise solidly-conservative GOP Senators who have decided the age of Trump is just too much for them.First Tennessee s Bob Corker then McCain s fellow Arizona Senator Jeff Flake dropped out of the running for reelection, possibly dashing the GOP s hopes for holding on to the Senate in the 2018 elections. But regardless of the fact that they ve both been what old Mitt Romney might call  severely conservative,  red meat Trump voters turned on the two like villagers with pitchforks.That could be, however, because they don t have the credentials John McCain has. When he trolled Trump hard last week on the  bone spurs  he used as an excuse to dodge the draft with all of his silver spoon, private school pals, Trump tried to clap back with a warning that  people have to be careful because at some point I fight back. McCain s answer? I have faced tougher adversaries. Now, you might think Donald Trump is at least smart enough not to come at a former POW with some weak stuff like that, but you d be wrong. Laughably wrong. In fact, go ahead and have a little laugh, because you re in good company.McCain thought it was hilarious.Appearing on The View the following day, the panel gave the senior Arizona Senator a chance to expound a little on his short response to the president s threat: He said he would  fight back  and it wouldn t be pretty. Are you scared? McCain didn t answer immediately because he was busy laughing his ass off. No, I mean, almost with tears in his eyes. There he is, next to his daughter   who just landed the spot on this show   and this old man looks like he s going to cry from laughing so hard. Every woman on the panel is cracking up. People offstage are howling.I just want to know Donald Trump has seen this clip. I can die happy as long as I know Donnie Daycare has watched this octogenarian cry tears of laughter at the prospect of being scared of little old him.Enjoy!McCain is asked if he's afraid of Trump. He laughs for about 15 straight second. pic.twitter.com/u7NYkkBGHf  Kyle Griffin (@kylegriffin1) October 23, 2017Featured image via Alex Wong/Getty Images",
            "WASHINGTON (Reuters) - Alabama Secretary of State John Merrill said he will certify Democratic Senator-elect Doug Jones as winner on Thursday despite opponent Roy Mooreâ€™s challenge, in a phone call on CNN. Moore, a conservative who had faced allegations of groping teenage girls when he was in his 30s, filed a court challenge late on Wednesday to the outcome of a U.S. Senate election he unexpectedly lost. ",
            " WATCH: Mike Penceâ€™s Photo Op With Puerto Rico Survivors Just Went TERRIBLY Wrong (VIDEO)",
            "The White House said on Friday it was set to kick off talks next week with Republican and Democratic congressional"
        
        ]
    )
    if st.button("Use sample text"):
        st.session_state['news_text'] = sample_news

st.title("📰 Fake News Classifier")
st.write("Paste a news article or headline below to check if it's Real or Fake.")
# Input box with session state for easy sample loading

news_text = st.text_area("Enter news text here:", value=st.session_state.get('news_text', ''), height=250)

if st.button("Classify"):
    if news_text.strip():
        label, confidence = predict_news(news_text)
        word_count, char_count, reading_time = get_text_stats(news_text)

        st.markdown(f"### Prediction: {'✅ Real News' if label=='Real News' else '❌ Fake News'}")
        st.markdown(f"**Confidence Score:** {confidence:.2%}")

        
             # Show text stats side-by-side
        col1, col2, col3 = st.columns([1,1,1])
        col1.metric("Word Count", word_count)
        col2.metric("Character Count", char_count)
        col3.metric("Estimated Reading Time", f"{reading_time:.2f} min")  
        
       

    else:
        st.warning("Please enter some text to classify.")

with st.expander("ℹ️ About this model and app"):
    st.write("""
    This model is trained using the Passive Aggressive Classifier algorithm, which is effective for large-scale text classification.
    
    It vectorizes news articles with TF-IDF, removing stopwords and non-alphabetical characters, then predicts if news is real or fake.
    
    The confidence score is calculated from the decision function of the classifier and converted to a probability-like score.
    """)

st.markdown("---")
st.markdown(
    "<center>Developed by Utsav </center>",
    unsafe_allow_html=True
)





