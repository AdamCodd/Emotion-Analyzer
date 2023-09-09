import csv
import json
import re
import os
import math
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class EmotionAnalyzer:
    DECAY_RATE = 0.15
    
    def __init__(self, lexicon_path: str, encoding='utf8', ambiguity_threshold=0.2):
        self.emotion_lexicon = self.load_emotion_lexicon(lexicon_path, encoding)
        self.sentence_splitter = re.compile(r'\.\.\.|[.?]')
        self.word_extractor = re.compile(r'\b(?:\w+\'\w+|\w+)[\.\?\!]*\b')
        self.PRE_CALCULATED_DECAYS = {distance: math.exp(-self.DECAY_RATE * distance) for distance in range(0, 5)}  
        self.BOOSTER_WEIGHTS = self._init_booster_weights()
        self.NEGATION_WEIGHTS = self._init_negation_weights()
        self.BOOSTER_AND_NEGATION_PHRASES = self.BOOSTER_WEIGHTS.keys() | self.NEGATION_WEIGHTS.keys()
        self.ambiguity_threshold = 0.1

    def _init_booster_weights(self) -> Dict[str, float]:
        booster_weights = {
        "extremely": 1.55,
        "absolutely": 1.45,
        "utterly": 1.45,
        "enormously": 1.45,
        "totally": 1.4,
        "immense": 1.4,
        "very": 1.4,
        "incredibly": 1.55,
        "highly": 1.35,
        "significantly": 1.35,
        "remarkably": 1.4,
        "truly": 1.25,
        "deeply": 1.3,
        "greatly": 1.3,
        "overwhelmingly": 1.5,
        "intensely": 1.4,
        "so much": 1.35,
        "a lot": 1.3,
        "cannot believe": 1.2,
        "can not believe": 1.2,
        "quite": 1.25,
        "never": 1.3,
        "so": 1.3,
        "really": 1.25,
        "trace": 1.1,
        "genuine": 1.1
        }
        return booster_weights
    
    def _init_negation_weights(self) -> Dict[str, float]:
        negation_weights  = {
        'no trace': 0.45,
        'put on': 0.7,
        'slight': 0.7,
        'slightly': 0.7,
        "do not": 0.6,
        "don't": 0.6,
        "don't be": 0.7,
        "do not be": 0.7,
        "didn't": 0.7,
        "isn't": 0.65,
        "is not": 0.65,
        "was not": 0.7,
        "wasn't": 0.7,
        "wouldn't": 0.7,
        "weren't": 0.7,
        "don't think": 0.7,
        "do not think": 0.7,
        "did not feel": 0.7,
        "impressive": 0.8,
        "without": 0.75,
        "feigning": 0.7,
        "hardly ever": 0.5,
        "any": 0.8,
        "anymore": 0.85,
        "forced": 0.7,
        "used to": 0.9,
        "kinda": 0.9,
        "not": 0.65,
        "yet": 0.9,
        "not really": 0.9,
        "barely": 0.5,
        "hardly": 0.5,
        "a bit": 0.8,
        "rarely": 0.6,
        "scarcely": 0.5
        }
        return negation_weights
    
    @staticmethod
    def load_emotion_lexicon(path: str, encoding: str = 'utf-8') -> Dict[str, Tuple[str, float]]:
        lexicon_dict = {}
        try:
            with open(path, newline='', encoding=encoding) as csvfile:
                reader = csv.reader(csvfile, delimiter=';')
                for row_number, row in enumerate(reader, start=1):
                    if len(row) == 3:
                        word, emotion, score = row
                        # Data Type Validation
                        if not isinstance(word, str) or not isinstance(emotion, str):
                            logging.warning(f"Row {row_number}: Invalid data type for word or emotion.")
                            continue
                        try:
                            score = float(score)
                        except ValueError:
                            logging.warning(f"Row {row_number}: Score is not a float.")
                            continue
                        # Value Range Checks
                        if not (0 <= score <= 1):
                            logging.warning(f"Row {row_number}: Score out of expected range [0, 1].")
                            continue
                        # Data Consistency
                        if word.lower() in lexicon_dict:
                            logging.warning(f"Row {row_number}: Duplicate entry for word '{word.lower()}'.")
                            continue

                        lexicon_dict[word.lower()] = (emotion, score)
                    else:
                        logging.info(f"Skipped malformed row {row_number}: {row}")
        except FileNotFoundError:
            logging.error("CSV file not found.")
            raise
        except Exception as e:
            logging.error(f"Error reading CSV file: {e}")
            raise
        return lexicon_dict
    
    def get_combined_modifier_weight(self, i: int, words: List[str], emotion_type_map: Dict[int, str], current_emotion: str) -> float:
        """
        Calculate the combined weight of booster and negation phrases around the word at index i.
        
        :param i: The index of the current word in the words list
        :param words: The list of words in the sentence
        :param emotion_type_map: A mapping of word indices to their associated emotion types
        :param current_emotion: The current emotion type being analyzed
        :return: The combined weight
        """
        closest_index, closest_emotion_type = min(
            [(k, v) for k, v in emotion_type_map.items() if i - 5 <= k <= i + 1],
            key=lambda x: abs(i - x[0]),
            default=(None, None)
        )
        if closest_emotion_type != current_emotion:
            return 1  # Default weight if no appropriate emotional word is found
        found_modifiers = []
        found_negatives = []
        for distance in range(1, 6):  # Check up to 5 words before and 1 word after
            for phrase in sorted(self.BOOSTER_AND_NEGATION_PHRASES, key=len, reverse=True):  # Sort by length
                phrase_words = phrase.split()
                comparison_words = [word.lower() for word in words[i - distance - len(phrase_words) + 1:i - distance + 1]]
                if comparison_words == phrase_words:
                    weight = self.BOOSTER_WEIGHTS.get(phrase, self.NEGATION_WEIGHTS.get(phrase))
                    decay = self.PRE_CALCULATED_DECAYS[abs(i - closest_index)]
                    if weight < 1:
                        found_negatives.append(weight * decay)
                    else:
                        found_modifiers.append(weight * decay)
        # If both negations and boosters are found, prioritize negations
        if found_negatives:
            return math.prod(found_negatives)
        if found_modifiers:
            return math.prod(found_modifiers)
        return 1  # Default weight if no booster/negation is found
    
    def calculate_weighted_score(self, original_score: float, i: int, words: List[str], emotion_type_map: Dict[int, str], emotion_type: str) -> float:
        """
        Calculate the weighted score based on the original score and other parameters.
        :param original_score: The original score of the word or phrase
        :param i: The index of the current word in the words list
        :param words: The list of words in the sentence
        :param emotion_type_map: A mapping of word indices to their associated emotion types
        :param emotion_type: The emotion type of the word or phrase
        :return: The weighted score
        """

        weight = self.get_combined_modifier_weight(i, words, emotion_type_map, emotion_type)
        weighted_score = original_score * weight
        weighted_score = min(1, max(0, weighted_score))
        return weighted_score
    
    def process_sentence(self, sentence: str) -> List[Tuple[str, str, float]]:
        """
        Process a single sentence to identify words that express emotions and their corresponding scores.        
        :return: A list of tuples, where each tuple contains the word, its emotion type, and its score
        """
        emotion_array = []
        words = self.word_extractor.findall(sentence)
        emotion_type_map = {}  # To store the index and emotion type of each detected emotion word
        i = 0
        while i < len(words):
            longest_match = 1  # Reset longest_match for each iteration
            matched = False

            # Check for phrases up to 4 words long
            for size in range(4, 0, -1):
                if i + size <= len(words):
                    phrase = ' '.join(words[i:i+size]).lower()
                    if phrase in self.emotion_lexicon:
                        emotion_type, score = self.emotion_lexicon[phrase]
                        emotion_type_map[i] = emotion_type  # Store the index and emotion type
                        
                        score = self.calculate_weighted_score(score, i, words, emotion_type_map, emotion_type)
                        emotion_array.append((phrase, emotion_type, score))
                        i += size  # Skip the words that made up the phrase
                        matched = True
                        break

            if not matched:
                # Check single words for emotion, booster, or negation.
                word = words[i].lower()
                if word in self.emotion_lexicon:
                    emotion_type, score = self.emotion_lexicon[word]
                    emotion_type_map[i] = emotion_type  # Store the index and emotion type
                    
                    score = self.calculate_weighted_score(score, i, words, emotion_type_map, emotion_type)
                    emotion_array.append((word, emotion_type, score))
                    
                i += 1  # Move to the next word

        return emotion_array
    
    def word_emotion_analysis(self, text: str, max_threads: int = 4) -> Tuple[str, float, float, List[Tuple[str, str, float]], Optional[Tuple[str, str]]]:
        """
        Perform emotion analysis on the given text.
        :return: A tuple containing the dominant emotion, its score, the highest word score,
                a list of emotion words and their scores, and any ambiguous emotions if present
        """
        # Split the text into individual sentences
        sentences = [sentence.strip() for sentence in self.sentence_splitter.split(text) if sentence]
        
        # Parallelize sentence processing using ThreadPoolExecutor
        actual_threads = min(max_threads, len(sentences))
        with ThreadPoolExecutor(max_workers=actual_threads) as executor:
            results = list(executor.map(self.process_sentence, sentences))
        
        # Flatten the list of lists into a single list containing all emotion words and their scores
        emotion_array = [item for sublist in results for item in sublist]

        # Handle cases where no emotion words are detected
        if not emotion_array:
            logging.info("No emotion word detected in the sentence. Default values used.")
            return 'None', 0, 0, [], ()

        # Calculate the cumulative scores for each emotion
        emotion_scores = defaultdict(list)
        for _, emotion, score in emotion_array:
            emotion_scores[emotion].append(score)

        # Calculate the overall dominant emotion and its score
        cumulative_scores = {emotion: sum(scores) for emotion, scores in emotion_scores.items()}
        highest_emotion = max(cumulative_scores, key=cumulative_scores.get)
        highest_scores = emotion_scores[highest_emotion]
        total_emotion_score = sum(cumulative_scores.values())
        
        # Calculate the raw score for the dominant emotion
        dominant_raw_score = sum(highest_scores) / len(highest_scores) if highest_scores else 0

        # Check if there are multiple emotions present
        if len(cumulative_scores) > 1:
            # Apply a moderated dampening effect to the dominant emotion's score based on other strong emotions
            other_emotion_score = total_emotion_score - cumulative_scores[highest_emotion]
            dampening_factor = min(other_emotion_score / total_emotion_score, 0.3)  # Cap the dampening at 30%
            highest_score = dominant_raw_score * (1 - dampening_factor)
        else:
            highest_score = dominant_raw_score  # No dampening needed if there's only one emotion

        # Identify any ambiguous emotions, if applicable
        sorted_emotions = sorted(cumulative_scores, key=cumulative_scores.get, reverse=True)
        confidence_wordanalysis = highest_score / total_emotion_score if total_emotion_score else None
        ambiguous_emotions = None
        if len(sorted_emotions) > 1 and confidence_wordanalysis < self.ambiguity_threshold:
            logging.info(f"Word analysis found ambiguous emotions between {sorted_emotions[0]} and {sorted_emotions[1]}")
            ambiguous_emotions = (sorted_emotions[0], sorted_emotions[1])
            
        # Calculate the highest word score within the dominant emotion
        highest_word_score = max(score for _, emotion, score in emotion_array if emotion == highest_emotion)

        return highest_emotion, highest_score, highest_word_score, emotion_array, ambiguous_emotions
    

class ResultsSaver:
    def __init__(self):
        self.analysis_results = []
        self.texts = []

    def read_input_file(self, input_file: str, json_key: str) -> List[str]:
        try:
            _, ext = os.path.splitext(input_file)
            format = ext.lstrip('.').lower()  # Remove the dot and convert to lowercase
            if format == 'txt':
                with open(input_file, 'r', encoding='utf-8') as f:
                    self.texts = f.readlines()
            elif format == 'csv':
                with open(input_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        self.texts.append(row[0])
            elif format == 'json':
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.texts = data.get(json_key, [])
            elif format == 'jsonl':
                with open(input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data = json.loads(line.strip())
                        if json_key:
                            self.texts.append(data.get(json_key, ''))
                        else:
                            self.texts.append(data)
            else:
                logging.error(f"Unsupported file format: {format}")
                return []
        except Exception as e:
            logging.error(f"An error occurred while reading the input file: {e}")
            return []
        return self.texts

    def save_results(self, results: Tuple[str, float, float, List[Tuple[str, str, float]], Optional[Tuple[str, str]]], text: str):
        dominant_emotion, highest_score, highest_word_score, emotion_array, ambiguous_emotions = results
        emotion_words_and_scores = [(word, emotion, round(score, 3)) for word, emotion, score in emotion_array]
        
        # Round the scores to 3 decimal places
        highest_score = round(highest_score, 3)
        highest_word_score = round(highest_word_score, 3)

        # Handle ambiguous emotions
        if ambiguous_emotions:
            dominant_emotion = "/".join(ambiguous_emotions)

        single_line_text = text.replace("\n", " ").replace("\r", "").strip()
        if single_line_text:  # Check if the text is not empty
            self.analysis_results.append({
                'Text': single_line_text,
                'Dominant Emotion': dominant_emotion,
                'Dominant Emotion Score': highest_score,
                'Highest Word Score': highest_word_score,
                'Words-Emotions-Scores': emotion_words_and_scores
            })

    @staticmethod
    def handle_file(file_path: str, mode: str, callback):
        with open(file_path, mode, encoding='utf-8') as f:
            callback(f)

    @staticmethod
    def csv_writer(f, data):
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(["Text", "Dominant Emotion", "Dominant Emotion Score", "Highest Word Score", "Emotion Words and Scores"])
        for result in data:
            writer.writerow([result['Text'], result['Dominant Emotion'], result['Dominant Emotion Score'], result['Highest Word Score'], result['Words-Emotions-Scores']])

    @staticmethod
    def txt_writer(f, data):
        for result in data:
            f.write(f"Text: {result['Text']}\n")
            f.write(f"Dominant Emotion: {result['Dominant Emotion']}\n")
            f.write(f"Dominant Emotion Score: {result['Dominant Emotion Score']}\n")
            f.write(f"Highest Word Score: {result['Highest Word Score']}\n")
            f.write(f"Words-Emotions-Scores: {result['Words-Emotions-Scores']}\n")
            f.write("\n")  # Separate entries

    @staticmethod
    def json_writer(f, data):
        json.dump(data, f, indent=4)

    @staticmethod
    def jsonl_writer(f, data):
        for result in data:
            if result['Text'].strip():  # Check if the 'Text' field is not empty
                f.write(json.dumps(result) + '\n')

    def save_results_generic(self, output_file: str, writer_func, data):
        self.handle_file(output_file, 'w', lambda f: writer_func(f, data))

    def save_results_to_csv(self, output_file: str, data):
        self.save_results_generic(output_file, self.csv_writer, data)

    def save_results_to_txt(self, output_file: str, data):
        self.save_results_generic(output_file, self.txt_writer, data)

    def save_results_to_json(self, output_file: str, data):
        self.save_results_generic(output_file, self.json_writer, data)

    def save_results_to_jsonl(self, output_file: str, data):
        self.save_results_generic(output_file, self.jsonl_writer, data)


def parse_args():
    parser = argparse.ArgumentParser(description='Perform emotion analysis.')
    parser.add_argument('--text', type=str, help='Text to analyze if not reading from a file')
    parser.add_argument('--lexicon_path', required=True, help='Path to the lexicon file')
    parser.add_argument('--max_threads', type=int, default=4, help='Maximum number of threads for parallel processing')
    parser.add_argument('--encoding', type=str, default='utf-8', help='File encoding for the lexicon (default is utf-8)')
    parser.add_argument('--ambiguity_threshold', type=float, default=0.1, help='Threshold for ambiguous emotions')
    parser.add_argument('--input_file', type=str, help='Input file to read texts from (csv, txt, json, jsonl supported)')
    parser.add_argument('--json_key', type=str, default='text', help='Key to read texts from when input file is JSON or JSONL')
    parser.add_argument('--output', type=str, default='output', help='Base name of the output file to save the results')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'txt', 'json', 'jsonl'], help='Output format: csv, txt, json, jsonl (default is csv)')
    return parser.parse_args()


# Main program
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = parse_args()
    
    # Initialize the EmotionAnalyzer with the lexicon path from command-line arguments
    emotion_analyzer = EmotionAnalyzer(args.lexicon_path, encoding=args.encoding, ambiguity_threshold=args.ambiguity_threshold)
    results_saver = ResultsSaver()

    if args.input_file:
        results_saver.texts = results_saver.read_input_file(args.input_file, args.json_key)

    elif args.text:
        # Use the text provided via command line
        results_saver.texts.append(args.text)
    else:
        # If no input is provided, use a sample text
        results_saver.texts.append("I am extremely happy, but really sad about the news.")

    for text in results_saver.texts:
        highest_emotion, highest_score, highest_word_score, emotion_array, ambiguous_emotions = emotion_analyzer.word_emotion_analysis(text, args.max_threads)
        
        results = (highest_emotion, highest_score, highest_word_score, emotion_array, ambiguous_emotions)
        
        # Append the results to the analysis_results list
        results_saver.save_results(results, text)

    # Save the results to a file
    if args.output:
        base_output_file, _ = os.path.splitext(args.output)
        output_file = f"{base_output_file}.{args.format}"

        if args.format == 'csv':
            results_saver.save_results_to_csv(output_file, results_saver.analysis_results)
        elif args.format == 'txt':
            results_saver.save_results_to_txt(output_file, results_saver.analysis_results)
        elif args.format == 'json':
            results_saver.save_results_to_json(output_file, results_saver.analysis_results)
        elif args.format == 'jsonl':
            results_saver.save_results_to_jsonl(output_file, results_saver.analysis_results)
        else:
            logging.error("Unsupported format.")

    # Output the results on the terminal if not reading from a file
    if not args.input_file:
        logging.info(f"\n Result of Emotion Analysis:")
        logging.info(f" Dominant Emotion: {highest_emotion}")
        logging.info(f" Dominant Emotion Score: {highest_score:.2f}")
        logging.info(f" Highest Word Score in Dominant Emotion: {highest_word_score:.2f}")

        if ambiguous_emotions:
            logging.info(f"\n Ambiguous Emotions Detected: {ambiguous_emotions[0]} and {ambiguous_emotions[1]}")

        logging.info("\n  Detailed Emotion Words and Scores:")
        for word, emotion, score in emotion_array:
            logging.info(f" Word: '{word}', Emotion: {emotion}, Score: {score:.2f}")
