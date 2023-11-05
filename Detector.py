import nltk
import textstat
import pickle
from Detector import runPrediction
nltk.download('punkt')
import math
import sklearn
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
device = "cpu"
model_id = "gpt2-large"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
def calculate_burstiness(numbers):
    """
    Calculate the standard deviation (burstiness) of a list of numbers.

    Parameters:
    - numbers: A list or array of numbers.

    Returns:
    - Standard deviation of the numbers.
    """
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    standard_deviation = math.sqrt(variance)
    return standard_deviation

def runPrediction(text):
    encodings = tokenizer(text, return_tensors="pt")
    per = calculate_perplexity(model, tokenizer, encodings, text)
    per["sentence"] = text
    return per

def calculate_perplexity(model, tokenizer, encodings, sentence, device="cpu", stride=512):
    # Calculate perplexity
    seq_len = encodings.input_ids.size(1)
    max_length = model.config.n_positions
    nlls = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return {"perplexity": ppl.item()}

def calculate_individual_burstiness(data_point, numbers):
    """
    Calculate the standard deviation of a single data point from the average of a list of numbers.

    Parameters:
    - data_point: The individual data point.
    - numbers: A list or array of numbers to calculate the average.

    Returns:
    - Standard deviation of the data point from the average.
    """
    n = len(numbers)
    mean = sum(numbers) / n
    variance = sum((x - mean) ** 2 for x in numbers) / n
    standard_deviation = math.sqrt(variance)
    data_point_deviation = math.fabs(data_point - mean)  # Absolute deviation of the data point from the mean
    return data_point_deviation

def evaluate_for_ai(text):
    print("Started")
    sentences = nltk.sent_tokenize(text)
    print("Tokenized")
    analyzedSentences = []
    perplexities = []
    perplexity = 0
    readability = 0
    #perform initial calculations for perplexity and readability
    print("Initial Calculations")
    i = 0
    for sentence in sentences:
        print(f"{i+1}/{len(sentences)} Complete")
        perp = runPrediction(sentence)
        read = textstat.automated_readability_index(sentence)
        perp["readability"] = read
        readability += read
        analyzedSentences.append(perp)
        perplexity += perp["perplexity"]
        perplexities.append(perp["perplexity"])
        i+= 1
    #calculate our average perplexity and overall burstiness
    print("calculating averages")
    averagePerp = perplexity/len(sentences)
    burstiness = calculate_burstiness(perplexities)
    #assign burstiness to each sentences
    print("Assigning burstiness")
    for analyzed in analyzedSentences:
        analyzed["burstiness"] = calculate_individual_burstiness(analyzed["perplexity"], perplexities)
        #load model
    
    #score each sentence with out model
    print("scoring")
    #load our model here
    with open('model-5.pkl', "rb") as model_file:
        loaded_model = pickle.load(model_file)
        predictions = 0
        for analyzed in analyzedSentences:
            pred = loaded_model.predict([(analyzed["perplexity"], analyzed["readability"], analyzed["burstiness"])])
            predictions += int(pred[0])
            analyzed["prediction"] = int(pred[0])
    results = {"average_generated_prob": predictions/len(sentences), "sentences": analyzedSentences, "overall_burstiness": burstiness, "average_perplexity": averagePerp, "average_readability": readability/len(sentences)}
    return results





