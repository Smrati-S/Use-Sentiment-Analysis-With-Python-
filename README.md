# Use-Sentiment-Analysis-With-Python-
Use Sentiment Analysis With Python to Classify Movie Reviews

Sentiment analysis is a powerful tool that allows computers to understand the underlying subjective tone of a piece of writing. This is something that humans have difficulty with, and as you might imagine, it isn’t always so easy for computers, either. But with the right tools and Python, you can use sentiment analysis to better understand the sentiment of a piece of writing.
Why would you want to do that? There are a lot of uses for sentiment analysis, such as understanding how stock traders feel about a particular company by using social media data or aggregating reviews.
Using Natural Language Processing to Pre-process and Clean Text Data-

import spacy
text = "I am learning sentiment analysis using nlp algorithms"
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
token_list
output- [ I,am,learning,sentiment,analysis,using,nlp,algorithms,]

Removing Stop Words

filtered_tokens = [token for token in doc if not token.is_stop]
filtered_tokens

output- [learning, sentiment, analysis, nlp, algorithms]

Lemmatization

lemmas = [ f”Token: {token}, lemma: {token.lemma_}”for token in filtered_tokens]
lemmas
output- ['Token: learning, lemma: learn',
'Token: sentiment, lemma: sentiment',
'Token: analysis, lemma: analysis',
'Token: nlp, lemma: nlp',
'Token: algorithms, lemma: algorithms']
Vectorizing Text
filtered_tokens[1].vector

array([ 2.2007031 ,  5.1163135 ,  0.3565974 , -0.42158726,  2.9876177 ,
       -1.981257  , -1.198557  ,  0.77857494, -1.5930126 ,  2.3974173 ,
        0.6465618 ,  1.3444257 , -1.1048071 , -0.17880744,  0.37357306,
       -0.56769204,  1.8486645 ,  1.1188302 , -2.8932462 ,  2.5493238 ,
       -0.9924354 ,  2.3645353 , -2.9235065 , -1.856147  , -1.3995124 ,
        0.93490183, -3.805128  , -3.70431   , -0.4328021 ,  0.6615139 ,
       -1.0992227 , -0.8818567 , -0.7660373 ,  0.29927173, -3.9941344 ,
        0.4578656 ,  2.8003364 ,  0.10503495,  0.20389064, -0.03172407,
        0.49987236,  0.77767813, -0.49859732, -0.3311656 ,  2.3649864 ,
       -0.9655422 , -0.5296351 ,  3.5888054 ,  0.32516515,  0.35143268,
       -1.4854883 , -1.8996634 , -1.7344913 , -1.6741319 ,  0.36769134,
        1.8666885 , -1.538296  ,  0.20801127, -1.3763142 ,  4.2677217 ,
       -0.08240402,  0.19832978, -0.72935617, -1.9397875 , -1.28541   ,1.7323294 , -0.42198214, -0.7754719 , -1.3384575 , -0.28999   ,
 -2.1255472 ,  1.256623  ,  2.379053  , -2.658495  ,  2.2833185 ,
 0.03698808,  0.20012069,  0.30576855, -3.1087668 , -1.4265417 ,
       
1.313834  , -0.6219284 ,  1.138799  ,  0.15824449, -0.7926483 ,
       -0.13726664,  0.30359578,  0.9322225 ,  2.5829644 , -1.1855149 ,
       -0.32142255,  1.7971289 ,  0.64198023,  0.02522269, -0.8814618 ,
        1.1769576 ], dtype=float32)

Using Machine Learning Classifiers to Predict Sentiment-

There are a number of tools available in Python for solving classification problems. Here are some of the more popular ones:
TensorFlow
PyTorch
scikit-learn
This list isn’t all-inclusive, but these are the more widely used machine learning frameworks available in Python. They’re large, powerful frameworks that take a lot of time to truly master and understand.

Building Your Own NLP Sentiment Analyzer
  Loading and Preprocessing Data-

def load_training_data(
    data_directory: str = "aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
 # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))

After loading the files, you want to shuffle them. This works to eliminate any possible bias from the order in which training data is loaded. Since the random module makes this easy to do in one line, you’ll also see how to split your shuffled data:
import os
import random

def load_training_data(
    data_directory: str = "aclImdb/train",
    split: float = 0.8,
    limit: int = 0
) -> tuple:
    # Load from files
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label}
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    split = int(len(reviews) * split)
    return reviews[:split], reviews[split:]

Here, you shuffle your data with a call to random.shuffle().

output-
(
'When tradition dictates that an artist must pass (...)',
{'cats': {'pos': True, 'neg': False}}
)

Once that’s done, you’ll be ready to build the training loop:

import os
import random
import spacy

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)

Next, you’ll handle the case in which the textcat component is present and then add the labels that will serve as the categories for your text:
import os
import random
import spacy

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")
Build Your Training Loop to Train textcat
import os
import random
import spacy
from spacy.util import minibatch, compounding
def train_model(
    training_data: list,
    test_data: list,
      iterations: int = 20
   ) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
       textcat = nlp.create_pipe(
                 "textcat", config={"architecture": "simple_cnn"}
      )
        nlp.add_pipe(textcat, last=True)
      else:
        textcat = nlp.get_pipe("textcat")

      textcat.add_label("pos")
      textcat.add_label("neg")

      # Train only textcat
      training_excluded_pipes = [
          pipe for pipe in nlp.pipe_names if pipe != "textcat"
      ]

Now you’re ready to add the code to begin training:

import os
import random
import spacy
from spacy.util import minibatch, compounding

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers

Here, you call nlp.begin_training(), which returns the initial optimizer function. This is what nlp.update() will use to update the weights of the underlying model.
Now you’ll begin training on batches of data:

import os
import random
import spacy
from spacy.util import minibatch, compounding

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )

For evaluate_model(), you’ll need to pass in the pipeline’s tokenizer component, the textcat component, and your test dataset:

def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Can't be 0 because of presence in denominator
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Every cats dictionary includes both labels. You can get all
            # the info you need with just the pos label.
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f-score": f_score}

You then use the score and true_label to determine true or false positives and true or false negatives. You then use those to calculate precision, recall, and f-score. Now all that’s left is to actually call evaluate_model():

def train_model(training_data: list, test_data: list, iterations: int = 20):
    # Previously seen code omitted for brevity.
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(
                    text,
                    labels,
                    drop=0.2,
                    sgd=optimizer,
                    losses=loss
                )
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

Once the training process is complete, it’s a good idea to save the model you just trained so that you can use it again without training a new model. After your training loop, add this code to save the trained model to a directory called model_artifacts located within your working directory:

# Save model
with nlp.use_params(optimizer.averages):
    nlp.to_disk("model_artifacts")

This snippet saves your model to a directory called model_artifacts so that you can make tweaks without retraining the model. Your final training function should look like this:

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Build pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn"}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Train only textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Training loop
        print("Beginning training")
        print("Loss\tPrecision\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # A generator that yields infinite series of input numbers
        for i in range(iterations):
            print(f"Training iteration {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                text, labels = zip(*batch)
                nlp.update(text, labels, drop=0.2, sgd=optimizer, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Save model
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")

Classifying Reviews

Here’s the test_model() signature along with the code to load your saved model:

def test_model(input_data: str=TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
import os
import random
import spacy
from spacy.util import minibatch, compounding

TEST_REVIEW = """
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)
"""

Next, you’ll pass this review into your model to generate a prediction, prepare it for display, and then display it to the user:

def test_model(input_data: str = TEST_REVIEW):
    #  Load saved trained model
    loaded_model = spacy.load("model_artifacts")
    # Generate prediction
    parsed_text = loaded_model(input_data)
    # Determine prediction to return
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Positive"
        score = parsed_text.cats["pos"]
    else:
        prediction = "Negative"
        score = parsed_text.cats["neg"]
    print(
        f"Review text: {input_data}\nPredicted sentiment: {prediction}"
        f"\tScore: {score}"
    )

In this code, you pass your input_data into your loaded_model, which generates a prediction in the cats attribute of the parsed_text variable. You then check the scores of each sentiment and save the highest one in the prediction variable.
You then save that sentiment’s score to the score variable. This will make it easier to create human-readable output, which is the last line of this function.
You’ve now written the load_data(), train_model(), evaluate_model(), and test_model() functions. That means it’s time to put them all together and train your first model.
Connecting the Pipeline

So far, you’ve built a number of independent functions that, taken together, will load data and train, evaluate, save, and test a sentiment analysis classifier in Python.
There’s one last step to make these functions usable, and that is to call them when the script is run. You’ll use the if __name__ == "__main__": idiom to accomplish this:

if __name__ == "__main__":
    train, test = load_training_data(limit=2500)
    train_model(train, test)
    print("Testing model")
    test_model()

Here you load your training data with the function you wrote in the Loading and Preprocessing Data section and limit the number of reviews used to 2500 total. You then train the model using the train_model() function you wrote in Training Your Classifier and, once that’s done, you call test_model() to test the performance of your model.
Note: With this number of training examples, training can take ten minutes or longer, depending on your system. You can reduce the training set size for a shorter training time, but you’ll risk having a less accurate model.
What did your model predict? Do you agree with the result? What happens if you increase or decrease the limit parameter when loading the data? Your scores and even your 
predictions may vary, but here’s what you should expect your output to look like:

$ python pipeline.py
Training model
Beginning training
Loss    Precision       Recall  F-score
11.293997120810673      0.7816593886121546      0.7584745762390477      0.7698924730851658
1.979159922178951       0.8083333332996527      0.8220338982702527      0.8151260503859189
[...]
0.000415042785704145    0.7926829267970453      0.8262711864056664      0.8091286306718204

Testing model

Review text:
Transcendently beautiful in moments outside the office, it seems almost
sitcom-like in those scenes. When Toni Colette walks out and ponders
life silently, it's gorgeous.<br /><br />The movie doesn't seem to decide
whether it's slapstick, farce, magical realism, or drama, but the best of it
doesn't matter. (The worst is sort of tedious - like Office Space with less humor.)

Predicted sentiment: Positive   Score: 0.8773064017295837
As your model trains, you’ll see the measures of loss, precision, and recall and the F-score for each training iteration. You should see the loss generally decrease. The precision, recall, and F-score will all bounce around, but ideally they’ll increase. Then you’ll see the test review, sentiment prediction, and the score of that prediction—the higher the better.

You’ve now trained your first sentiment analysis machine learning model using natural language processing techniques and neural networks with spaCy! Here are two charts showing the model’s performance across twenty training iterations. The first chart shows how the loss changes over the course of training:

While the above graph shows loss over time, the below chart plots the precision, recall, and F-score over the same training period:

In these charts, you can see that the loss starts high but drops very quickly over training iterations. The precision, recall, and F-score are pretty stable after the first few training iterations. What could you tinker with to improve these values?

Conclusion

Congratulations on building your first sentiment analysis model in Python! What did you think of this project? Not only did you build a useful tool for data analysis, but you also picked up on a lot of the fundamental concepts of natural language processing and machine learning.
Next Steps With Sentiment Analysis and Python
This is a core project that, depending on your interests, you can build a lot of functionality around. Here are a few ideas to get you started on extending this project:
The data-loading process loads every review into memory during load_data(). Can you make it more memory efficient by using generator functions instead?
Rewrite your code to remove stop words during preprocessing or data loading. How does the mode performance change? Can you incorporate this preprocessing into a pipeline component instead?
Use a tool like Click to generate an interactive command-line interface.

Deploy your model to a cloud platform like AWS and wire an API to it. This can form the basis of a web-based tool.
Explore the configuration parameters for the textcat pipeline component and experiment with different configurations.
Explore different ways to pass in new reviews to generate predictions.
Parametrize options such as where to save and load trained models, whether to skip training or train a new model, and so on.
This project uses the Large Movie Review Dataset, which is maintained by Andrew Maas. Thanks to Andrew for making this curated dataset widely available for use.






