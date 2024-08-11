import os
import pandas as pd
import random
import ast
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from transformers import AutoTokenizer
from evaluation import *
from sklearn.metrics import (
    precision_score, recall_score, f1_score, hamming_loss, jaccard_score,
    log_loss, average_precision_score
)
from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets, evaluation
from torch.utils.data import DataLoader
from data import DATA_DIR
from utils.constants import NON_OVERLAPPING_SDGS
from warnings import simplefilter
from sklearn.model_selection import train_test_split
from arg_parser import get_args

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

class DescriptionDataLoader:
    def __init__(self, label_desc_dir, sbert_model):
        self.label_desc_dir = label_desc_dir
        self.sbert_model = sbert_model
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(ENGLISH_STOP_WORDS)
        self.sdg_definitions = self.get_sdg_definition()

    def read_descriptions(self, file_path):
        df = pd.read_csv(file_path, sep='\t', encoding='utf-8')
        return df['description'].tolist()

    def get_sdg_definition(self):
        sdg_definitions = {}
        for sdg_folder in os.listdir(self.label_desc_dir):
            sdg_path = os.path.join(self.label_desc_dir, sdg_folder)
            if os.path.isdir(sdg_path) and not sdg_folder.startswith('.'):
                sdg_definitions[sdg_folder] = {"Goal": [], "Targets": [], "Indicators": []}
                for category in sdg_definitions[sdg_folder].keys():
                    file_path = os.path.join(sdg_path, f'{category.lower()}.txt')
                    if os.path.exists(file_path) and file_path.endswith('.txt'):
                        descriptions = self.read_descriptions(file_path)
                        sdg_definitions[sdg_folder][category].extend(descriptions)
        return sdg_definitions



class MultiLabelDatasetLoader:
    def __init__(self, multi_label_data_dir):
        self.multi_label_data_dir = multi_label_data_dir
        #self.sbert_model = sbert_model


    def create_labels(self, row):
        labels = []
        for i in range(1, 18):
            col_name = f'SDG-{i:02d}'
            if row[col_name] == 1:
                labels.append(col_name.replace('-', ''))
        return labels

    def read_dataset(self):

        df = pd.read_csv(os.path.join(self.multi_label_data_dir, 'sdg_knowledge_hub.csv'), encoding='utf-8',
                         engine='python')

        df['labels'] = df.apply(self.create_labels, axis=1)

        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        return train_df, test_df

class DescriptionFineTuning:
    def __init__(self, sdg_definitions, non_overlapping_sdgs, sbert_model):
        self.sdg_definitions = sdg_definitions
        self.non_overlapping_sdgs = non_overlapping_sdgs
        self.sbert_model = sbert_model

    def sentence_pairs_generation(self, sdg):
        pairs = []
        goal = self.sdg_definitions[sdg]['Goal'][0]
        positives = self.sdg_definitions[sdg]['Targets']
        # Generating positive pairs
        for pos in positives:
            pairs.append(InputExample(texts=[goal, pos], label=1.0))

        # Number of positive pairs
        num_positives = len(positives)
        # Generating negative pairs
        negative_samples = []
        non_overlapping = self.non_overlapping_sdgs.get(sdg, [])
        for neg_sdg in non_overlapping:
            if neg_sdg in self.sdg_definitions:
                negatives = self.sdg_definitions[neg_sdg]['Targets']
                for neg in negatives:
                    negative_samples.append(neg)

        sampled_negatives = random.sample(negative_samples, num_positives)

        for neg in sampled_negatives:
            pairs.append(InputExample(texts=[goal, neg], label=0.0))

        return pairs

    def prepare_data(self):
        train_examples = []
        for sdg in self.sdg_definitions.keys():
            sdg_pairs = self.sentence_pairs_generation(sdg)
            train_examples.extend(sdg_pairs)
        return train_examples

    def train_model(self, train_examples):
        random.shuffle(train_examples)
        # S-BERT adaptation
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10, show_progress_bar=True)
        return self.sbert_model


class MultiLabelSBERTFineTuning:
    def __init__(self, train_df, sbert_model):
        self.train_df = train_df
        #self.test_df = test_df
        self.sbert_model = sbert_model
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
        self.text_col = self.train_df.columns.values[3]
        self.labels_col = self.train_df.columns.values[23]

        # self.text_col = self.train_df.columns.values[3]
        # self.labels_col = self.train_df.columns.values[2]

    def get_unique_labels(self, y_train):
        unique_labels = set()
        for labels in y_train:
            unique_labels.update(labels)
        return list(unique_labels)

    def sentence_pairs_generation(self, sentences, labels, pairs):

        label_sets = [set(label) for label in labels]
        for idxA in range(len(sentences)):
            current_sentence = sentences[idxA]
            label = label_sets[idxA]
            pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set == label and idx != idxA]
            if not pos_indices:

                pos_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set & label and idx != idxA]
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                #label_pos = label_sets[idxB]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

                if not pos_indices:
                    print("No positive sentence found.\n")
                    continue

            else:
                pos_idx = np.random.choice(pos_indices)
                pos_sentence = sentences[pos_idx]
                pairs.append(InputExample(texts=[current_sentence, pos_sentence], label=1.0))

            neg_indices = [idx for idx, lbl_set in enumerate(label_sets) if lbl_set != label and idx != idxA]
            if not neg_indices:
                print("No negative sentence found.\n")
                continue

            else:
                neg_idx = np.random.choice(neg_indices)
                neg_sentence = sentences[neg_idx]
                pairs.append(InputExample(texts=[current_sentence, neg_sentence], label=0.0))

        return (pairs)

    def sample_train_data(self, args):

        train_positive = []
        for sdg in range(1, 18):
            sdg_str = f'SDG{str(sdg).zfill(2)}'
            positive_df = self.train_df[self.train_df['labels'].apply(lambda labels: sdg_str in labels and len(labels) <= 10)]
            #sdg_negative_df = self.train_df[(self.train_df[sdgs] == sdg) & (self.train_df[labels] == False)]

            if len(positive_df) >= args.num_training:
                train_positive.append(positive_df.sample(n=args.num_training, random_state=args.seed))

        train_df_sample = pd.concat(train_positive).reset_index(drop=True)
        # train_df_negative_sample = pd.concat(balanced_train_negative).reset_index(drop=True)
        x_train = train_df_sample[self.text_col].values.tolist()
        y_train = train_df_sample[self.labels_col].values.tolist()

        return x_train, y_train

    def prepare_data(self):

        train_examples = []
        args = get_args()
        x_train, y_train = self.sample_train_data(args)
        for x in range(args.num_iter):
            train_examples = self.sentence_pairs_generation(x_train, y_train, train_examples)

        return x_train, y_train, train_examples

    def head_tail_truncation(self, text, max_length=512, head_tokens=128, tail_tokens=382):
        """
        Truncates the input text, keeping the first `head_tokens` and the last `tail_tokens` tokens,
        ensuring the total length does not exceed `max_length`.
        """
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) > max_length:
            if head_tokens + tail_tokens > max_length:
                raise ValueError("Sum of head_tokens and tail_tokens exceeds max_length.")
            truncated_tokens = tokens[:head_tokens] + tokens[-tail_tokens:]
        else:
            truncated_tokens = tokens

        return self.tokenizer.convert_tokens_to_string(truncated_tokens)

    def sbert_finetuning(self, train_examples):

        random.shuffle(train_examples)
        truncated_train_examples = []
        for example in train_examples:
            truncated_texts = [
                self.head_tail_truncation(text) for text in example.texts
            ]
            truncated_train_examples.append(InputExample(texts=truncated_texts, label=example.label))

        # S-BERT adaptation
        #train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
        train_dataloader = DataLoader(truncated_train_examples, shuffle=True, batch_size=16)
        train_loss = losses.CosineSimilarityLoss(self.sbert_model)
        self.sbert_model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10,
                                     show_progress_bar=True)

        return self.sbert_model

class LinearClassifier:

    def __init__(self, test_df):
        self.linear_classifier = OneVsRestClassifier(LogisticRegression())
        self.mlb = MultiLabelBinarizer()
        self.test_df = test_df
        self.text_col = self.test_df.columns.values[3]
        self.labels_col = self.test_df.columns.values[23]


    def train_model(self, x_train, y_train, model):

        x_eval = self.test_df[self.text_col].values.tolist()
        y_train_binary = self.mlb.fit_transform(y_train)

        X_train = np.array(model.encode(x_train))
        X_eval = np.array(model.encode(x_eval))

        y_train_binary = np.array(y_train_binary)
        self.linear_classifier.fit(X_train, y_train_binary)
        return X_eval, self.linear_classifier

    def predict(self, X_eval, threshold=0.5):
        # Predict probabilities
        y_pred_proba = self.linear_classifier.predict_proba(X_eval)

        # Apply threshold to get binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Convert binary predictions back to labels
        predicted_labels = self.mlb.inverse_transform(y_pred)

        return predicted_labels, y_pred_proba, y_pred

    def eval_model(self, X_eval, classifier):
        """
        Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
        :param X_eval: The input features for evaluation.
        :param classifier: The trained classifier model.
        :return: A tuple containing various evaluation metrics.
        """

        threshold = 0.5
        y_eval = self.test_df[self.labels_col].values.tolist()
        y_eval_binary = self.mlb.transform(y_eval)

        # Predict probabilities
        y_pred_proba = classifier.predict_proba(X_eval)

        # Apply threshold to probabilities to obtain binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Compute strong accuracy
        strict_matches = np.all(y_pred == y_eval_binary, axis=1)
        strict_accuracy = np.mean(strict_matches)

        # Compute weak accuracy
        weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
        weak_accuracy = np.mean(weak_matches)

        # Compute per-class metrics
        num_classes = y_eval_binary.shape[1]
        APs = []
        for j in range(num_classes):
            AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
            APs.append(AP)

        # Compute macro mAP (unweighted average of APs)
        macro_mAP = np.mean(APs)

        # Compute weighted mAP (average of APs weighted by number of positives)
        class_counts = np.sum(y_eval_binary, axis=0)
        weighted_mAP = np.average(APs, weights=class_counts)

        # Compute micro mAP (global-based)
        micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())

        # Compute Hamming loss
        hamming = hamming_loss(y_eval_binary, y_pred)

        # Compute Precision, Recall, F1-Score
        precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
        recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
        precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
        recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
        f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
        f1_macro = f1_score(y_eval_binary, y_pred, average='macro')

        # Compute Jaccard Similarity
        jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')

        # Compute Log Loss
        logloss = log_loss(y_eval_binary, y_pred_proba)

        # Return all the computed metrics
        return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
                recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs)



class LinearClassifierORO:

    def __init__(self, test_df):
        self.linear_classifier = OneVsRestClassifier(LogisticRegression())
        self.mlb = MultiLabelBinarizer()
        self.test_df = test_df
        self.text_col = self.test_df.columns.values[7]
        self.labels_col = self.test_df.columns.values[6]



    def train_model(self, x_train, y_train, model):

        x_eval = self.test_df[self.text_col].values.tolist()

        y_train_binary = self.mlb.fit_transform(y_train)

        X_train = np.array(model.encode(x_train))
        X_eval = np.array(model.encode(x_eval))

        y_train_binary = np.array(y_train_binary)
        #self.linear_classifier.fit(X_train, y_train)
        self.linear_classifier.fit(X_train, y_train_binary)
        return X_eval, self.linear_classifier



    def eval_model(self, X_eval, classifier):
        """
        Given the predictions and golds, run the evaluation in several modes: weak, strict, ...
        :param X_eval: The input features for evaluation.
        :param classifier: The trained classifier model.
        :return: A tuple containing various evaluation metrics.
        """

        threshold = 0.5
        y_eval = self.test_df[self.labels_col].values.tolist()
        y_eval_binary = self.mlb.transform(y_eval)

        # Predict probabilities
        y_pred_proba = classifier.predict_proba(X_eval)

        # Apply threshold to probabilities to obtain binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Compute strong accuracy
        strict_matches = np.all(y_pred == y_eval_binary, axis=1)
        strict_accuracy = np.mean(strict_matches)

        # Compute weak accuracy
        weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
        weak_accuracy = np.mean(weak_matches)

        # Compute per-class metrics
        num_classes = y_eval_binary.shape[1]
        APs = []
        for j in range(num_classes):
            AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
            APs.append(AP)

        # Compute macro mAP (unweighted average of APs)
        macro_mAP = np.mean(APs)

        # Compute weighted mAP (average of APs weighted by number of positives)
        class_counts = np.sum(y_eval_binary, axis=0)
        weighted_mAP = np.average(APs, weights=class_counts)

        # Compute micro mAP (global-based)
        micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())

        # Compute Hamming loss
        hamming = hamming_loss(y_eval_binary, y_pred)

        # Compute Precision, Recall, F1-Score
        precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
        recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
        precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
        recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
        f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
        f1_macro = f1_score(y_eval_binary, y_pred, average='macro')

        # Compute Jaccard Similarity
        jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')

        # Compute Log Loss
        logloss = log_loss(y_eval_binary, y_pred_proba)

        # Return all the computed metrics
        return (strict_accuracy, weak_accuracy, hamming, precision_micro, recall_micro, f1_micro, precision_macro,
                recall_macro, f1_macro, jaccard_sim, logloss, macro_mAP, weighted_mAP, micro_mAP, APs)

    def predict(self, X_eval, threshold=0.5):
        # Predict probabilities
        y_pred_proba = self.linear_classifier.predict_proba(X_eval)

        # Apply threshold to get binary predictions
        y_pred = (y_pred_proba >= threshold).astype(int)

        # Convert binary predictions back to labels
        predicted_labels = self.mlb.inverse_transform(y_pred)

        return predicted_labels, y_pred_proba, y_pred

    def evaluate_metrics(self, y_eval, y_pred, y_pred_proba):
        # Handle empty values (consider filtering or imputation)
        valid_indices = [i for i, row in enumerate(y_eval) if any(row)]  # Indices with non-empty labels
        y_eval = np.array(y_eval)[valid_indices]
        y_pred = y_pred[valid_indices]
        y_pred_proba = y_pred_proba[valid_indices]

        # Multi-label binarization (assuming self.mlb is already fit)
        y_eval_binary = self.mlb.transform(y_eval)

        # Compute metrics
        strict_matches = np.all(y_pred == y_eval_binary, axis=1)
        strict_accuracy = np.mean(strict_matches)

        weak_matches = np.any(np.logical_and(y_pred, y_eval_binary), axis=1)
        weak_accuracy = np.mean(weak_matches)

        num_classes = y_eval_binary.shape[1]
        APs = []
        class_counts = np.zeros(num_classes)  # Initialize class counts with zeros
        for j in range(num_classes):
            # Check for empty class in ground truth
            if not np.any(y_eval_binary[:, j]):
                APs.append(0)  # Assign 0 AP for empty class
            else:
                # Calculate AP for non-empty class
                AP = average_precision_score(y_eval_binary[:, j], y_pred_proba[:, j])
                APs.append(AP)
                class_counts[j] = np.sum(y_eval_binary[:, j])  # Update class count

        # Compute mAPs
        macro_mAP = np.mean(APs)
        weighted_mAP = np.average(APs, weights=class_counts)
        micro_mAP = average_precision_score(y_eval_binary.ravel(), y_pred_proba.ravel())

        # Other metrics
        hamming = hamming_loss(y_eval_binary, y_pred)
        precision_micro = precision_score(y_eval_binary, y_pred, average='micro')
        recall_micro = recall_score(y_eval_binary, y_pred, average='micro')
        f1_micro = f1_score(y_eval_binary, y_pred, average='micro')
        precision_macro = precision_score(y_eval_binary, y_pred, average='macro')
        recall_macro = recall_score(y_eval_binary, y_pred, average='macro')
        f1_macro = f1_score(y_eval_binary, y_pred, average='macro')
        jaccard_sim = jaccard_score(y_eval_binary, y_pred, average='micro')
        logloss = log_loss(y_eval_binary, y_pred_proba)

        # Return dictionary of metrics
        metrics = {
            "Strict Accuracy": strict_accuracy,
            "Weak Accuracy": weak_accuracy,
            "Macro mAP": macro_mAP,
            "Weighted mAP": weighted_mAP,
            "Micro mAP": micro_mAP,
            "Hamming Loss": hamming,
            "Precision (Micro)": precision_micro,
            "Recall (Micro)": recall_micro,
            "F1 Score (Micro)": f1_micro,
            "Precision (Macro)": precision_macro,
            "Recall (Macro)": recall_macro,
            "F1 Score (Macro)": f1_macro,
            "Jaccard Similarity": jaccard_sim,
            "Log Loss": logloss
        }

        return metrics


def desc_finetuning(model):
    LABEL_DESC_DIR = os.path.join(DATA_DIR, 'label_desc')

    sdg_data_loader = DescriptionDataLoader(LABEL_DESC_DIR, model)
    sdg_trainer = DescriptionFineTuning(sdg_data_loader.sdg_definitions, NON_OVERLAPPING_SDGS, model)

    train_examples = sdg_trainer.prepare_data()
    model = sdg_trainer.train_model(train_examples)
    return model

def multi_label_classification(model):
    args = get_args()
    MULTI_LABEL_DATA_DIR = os.path.join(DATA_DIR, 'sdg_knowledge_hub')
    multi_label_data_loader = MultiLabelDatasetLoader(MULTI_LABEL_DATA_DIR)
    train_df, test_df = multi_label_data_loader.read_dataset()

    sbert_finetuning = MultiLabelSBERTFineTuning(train_df, model)
    x_train, y_train, train_examples = sbert_finetuning.prepare_data()
    if args.multi_label_finetuning:
        embedding_model = sbert_finetuning.sbert_finetuning(train_examples)
    else:
        embedding_model = model
    multilabel_sdg_trainer = LinearClassifier(test_df)
    X_eval, classifier = multilabel_sdg_trainer.train_model(x_train, y_train, embedding_model)
    results = multilabel_sdg_trainer.eval_model(X_eval, classifier)

    return results









