import json
import string
import aspell
import errant
import random
import argparse
import numpy as np

from src.utils.edit import Edit
from src.utils.errors import ERRORS
from src.utils.MorphoDiTa.generate_forms import GenerateForms

from typing import List
from typing import Tuple
from typing import Optional
from itertools import compress
from importlib import resources
from errant.annotator import Annotator


allowed_source_delete_tokens = [',', '.', '!', '?']
czech_diacritics_tuples = [('a', 'á'), ('c', 'č'), ('d', 'ď'), ('e', 'é', 'ě'), ('i', 'í'), ('n', 'ň'), ('o', 'ó'), ('r', 'ř'), ('s', 'š'),
                           ('t', 'ť'), ('u', 'ů', 'ú'), ('y', 'ý'), ('z', 'ž')]
czech_diacritizables_chars = [char for sublist in czech_diacritics_tuples for char in sublist] + [char.upper() for sublist in
                                                                                                  czech_diacritics_tuples for char in
                                                                                                  sublist]

random.seed(42)

# MAIN:
class ErrorGenerator:
    def __init__(self, config: dict, 
                 word_vocabulary, char_vocabulary,
                 char_err_distribution, char_err_prob, char_err_std,
                 token_err_distribution, token_err_prob, token_err_std,
                 derinet_distance = 0) -> None:
        self.inner_iterator = 0
        self.annotator = None
        self.total_tokens = 0

        self.char_err_distribution = char_err_distribution
        self.char_err_prob = char_err_prob
        self.char_err_std = char_err_std
        self.char_vocabulary = char_vocabulary

        self.token_err_distribution = token_err_distribution
        self.token_err_prob = token_err_prob
        self.token_err_std = token_err_std
        self.word_vocabulary = word_vocabulary

        self.derinet_distance = derinet_distance

        self.error_instances = []
        for error_name, [prob_per_tok, abs_prob, use_abs] in config.items():
            self.error_instances.append(ERRORS[error_name](prob_per_tok, abs_prob, use_abs))

    def _init_annotator(self, lang: str = 'cs'):
        if self.annotator is None:
            self.annotator = errant.load(lang)

    def get_edits(self, parsed_sentence, annotator: Annotator, aspell_speller,
                  count_output: Optional[str] = None, window: int = 200) -> List[Edit]:
        self.inner_iterator += 1
        self.total_tokens += len(parsed_sentence)
        edits_errors = []
        for error_instance in self.error_instances:
            edits = error_instance(parsed_sentence, annotator, aspell_speller)
            edits_errors = edits_errors + [(edit, error_instance) for edit in edits]
        
        if len(edits_errors) == 0:
            return []

        # Overlaping:
        random.shuffle(edits_errors)
        mask = self.get_remove_mask(list(zip(*edits_errors))[0])
        edits_errors = list(compress(edits_errors, mask))
        
        selected_edits = []
        ## Rejection Sampling:
        for edit, error_instance in edits_errors:
            if not error_instance.use_absolute_prob:
                acceptance_prob = (((1.0 * error_instance.target_prob) * (self.total_tokens + window)) - (error_instance.num_errors * 1.0)) / window
                if np.random.uniform(0, 1) < acceptance_prob:
                    selected_edits.append(edit)
                    error_instance.num_errors += 1
                error_instance.num_possible_edits += 1
        ##

        ## Absolute probability:
        for edit, error_instance in edits_errors:
            if error_instance.use_absolute_prob:
                if np.random.uniform(0, 1) < error_instance.absolute_prob:
                    selected_edits.append(edit)
                    error_instance.num_errors += 1
                error_instance.num_possible_edits += 1
        ##
            
        ## Write counts:
        if count_output:
            if self.inner_iterator % 100 == 0:
                with open(count_output, "w") as file:
                    file.write("Counts:\n")
                    for error_instance in self.error_instances:
                        error_name = error_instance.__class__.__name__
                        pos_errors = error_instance.num_possible_edits
                        num_errors = error_instance.num_errors
                        abs_prob = round(num_errors / (pos_errors + 1e-10), 2)
                        ratio = round(pos_errors / (self.total_tokens + 1e-10), 5)
                        file.write(error_name + '\n' + \
                                   str(abs_prob) + '\t' + str(pos_errors) + '\t' + str(num_errors) + '\t' + \
                                   str(self.total_tokens) + "\t" + str(ratio) + "\n")
        ##

        # Sorting:
        sorted_edits = self.sort_edits(selected_edits, True)
        return sorted_edits
    
    @staticmethod
    def sort_edits(edits: List[Edit], reverse: bool = False) -> List[Edit]:
        reverse_index = -1 if reverse else 1
        minus_start_indices = [reverse_index * edit.o_end for edit in edits]
        sorted_edits = np.array(edits)
        sorted_edits = sorted_edits[np.argsort(minus_start_indices)]

        minus_start_indices = [reverse_index * edit.o_start for edit in sorted_edits]
        sorted_edits = np.array(sorted_edits)
        sorted_edits = sorted_edits[np.argsort(minus_start_indices)]

        return sorted_edits.tolist()

    @staticmethod
    def get_remove_mask(edits: List[Edit]) -> List[bool]:
        ranges = [(edit.o_start, edit.o_end) for edit in edits]
        removed = [not any([ErrorGenerator.is_overlap(current_range, r) if j < i else False for j, r in enumerate(ranges)]) for i, current_range in enumerate(ranges)]
        # filtered_edits = list(compress(edits, removed))
        return removed

    @staticmethod
    def is_overlap(range_1: tuple, range_2: tuple) -> bool:
        start_1 = range_1[0]
        end_1 = range_1[1]
        start_2 = range_2[0]
        end_2 = range_2[1]

        if start_1 <= start_2:
            if end_1 > start_2:
                return True
        else:
            if end_2 > start_1:
                return True
        return False
    
    def turn_edits(self, parsed_sentence, edits: List[Edit]) -> Tuple[str, List[Edit]]:
        sentence = self._use_edits(edits, parsed_sentence)
        turn_edits = []
        for edit in edits:
            turn_edit = Edit(edit.c_toks, edit.o_toks, [edit.c_start, edit.c_end, edit.o_start, edit.o_end], type=edit.type)
            turn_edits.append(turn_edit)
        return sentence, turn_edits

    def get_m2_edits_text(self, sentence: str, aspell_speller) -> Tuple[str, List[str]]:
        parsed_sentence = self.annotator.parse(sentence)
        error_edits = self.get_edits(parsed_sentence, self.annotator, aspell_speller)
        error_sentence, edits = self.turn_edits(parsed_sentence, error_edits)
        edits = self.sort_edits(edits)
        m2_edits = [edit.to_m2() for edit in edits]
        return error_sentence, m2_edits
    
    def introduce_token_level_errors_on_sentence(self, tokens, aspell_speller = None, morphodita = None):
        num_errors = int(np.round(np.random.normal(self.token_err_prob, self.token_err_std) * len(tokens)))
        num_errors = min(max(0, num_errors), len(tokens)) 

        if num_errors == 0:
            return ' '.join(tokens)
        token_ids_to_modify = np.random.choice(len(tokens), num_errors, replace=False)

        new_sentence = ''
        for token_id in range(len(tokens)):
            if token_id not in token_ids_to_modify:
                if new_sentence:
                    new_sentence += ' '
                new_sentence += tokens[token_id]
                continue

            current_token = tokens[token_id]
            operation = np.random.choice(['replace_aspell', 'replace_morphodita', 'insert', 'delete', 'swap', 'recase'], p=self.token_err_distribution)
            new_token = ''
            if operation == 'replace_aspell':
                if not aspell_speller:
                    raise Exception("Aspell speller is not loaded...")
                if not current_token.isalpha():
                    new_token = current_token
                else:
                    proposals = aspell_speller.suggest(current_token)[:10]
                    if len(proposals) > 0:
                        new_token = np.random.choice(proposals) 
                    else:
                        new_token = current_token
            elif operation == 'replace_morphodita':
                if not morphodita:
                    raise Exception("Morphodita is not loaded...")
                if not current_token.isalpha():
                    new_token = current_token
                else:
                    proposals = list(morphodita.forms(current_token, self.derinet_distance))
                    proposals = proposals[:10]
                    if len(proposals) > 0:
                        new_token = np.random.choice(proposals)
                    else:
                        new_token = current_token
            elif operation == 'insert':
                new_token = current_token + ' ' + np.random.choice(self.word_vocabulary)
            elif operation == 'delete':
                if not current_token.isalpha() or current_token in allowed_source_delete_tokens:
                    new_token = current_token
                else:
                    new_token = ''
            elif operation == 'recase':
                if not current_token.isalpha():
                    new_token = current_token
                elif current_token.islower():
                    new_token = current_token[0].upper() + current_token[1:]
                else:
                    # either whole word is upper-case or mixed-case
                    if np.random.random() < 0.5:
                        new_token = current_token.lower()
                    else:
                        num_recase = min(len(current_token), max(1, int(np.round(np.random.normal(0.3, 0.4) * len(current_token)))))
                        char_ids_to_recase = np.random.choice(len(current_token), num_recase, replace=False)
                        new_token = ''
                        for char_i, char in enumerate(current_token):
                            if char_i in char_ids_to_recase:
                                if char.isupper():
                                    new_token += char.lower()
                                else:
                                    new_token += char.upper()
                            else:
                                new_token += char

            elif operation == 'swap':
                if token_id == len(tokens) - 1:
                    continue

                new_token = tokens[token_id + 1]
                tokens[token_id + 1] = tokens[token_id]

            if new_sentence and new_token:
                new_sentence += ' '
            new_sentence = new_sentence + new_token

        return new_sentence
    
    def introduce_char_level_errors_on_sentence(self, sentence):
        sentence = list(sentence)
        num_errors = int(np.round(np.random.normal(self.char_err_prob, self.char_err_std) * len(sentence)))
        num_errors = min(max(0, num_errors), len(sentence)) 
        if num_errors == 0:
            return ''.join(sentence)
        char_ids_to_modify = np.random.choice(len(sentence), num_errors, replace=False)
        new_sentence = ''
        for char_id in range(len(sentence)):
            if char_id not in char_ids_to_modify:
                new_sentence += sentence[char_id]
                continue
            operation = np.random.choice(['replace', 'insert', 'delete', 'swap', 'change_diacritics'], 1,
                                         p=self.char_err_distribution)
            current_char = sentence[char_id]
            new_char = ''
            if operation == 'replace':
                if current_char.isalpha():
                    new_char = np.random.choice(self.char_vocabulary)
                else:
                    new_char = current_char
            elif operation == 'insert':
                new_char = current_char + np.random.choice(self.char_vocabulary)
            elif operation == 'delete':
                if current_char.isalpha():
                    new_char = ''
                else:
                    new_char = current_char
            elif operation == 'swap':
                if char_id == len(sentence) - 1:
                    continue
                new_char = sentence[char_id + 1]
                sentence[char_id + 1] = sentence[char_id]
            elif operation == 'change_diacritics':
                if current_char in czech_diacritizables_chars:
                    is_lower = current_char.islower()
                    current_char = current_char.lower()
                    char_diacr_group = [group for group in czech_diacritics_tuples if current_char in group][0]
                    new_char = np.random.choice(char_diacr_group)
                    if not is_lower:
                        new_char = new_char.upper()
            new_sentence += new_char
        return new_sentence
    
    def create_error_sentence(self, sentence: str, aspell_speller, 
                              use_token_level: bool = False, use_char_level: bool = False, 
                              morfodita=None, count_output: Optional[str] = None) -> List[str]:
        parsed_sentence = self.annotator.parse(sentence)
        edits = self.get_edits(parsed_sentence, self.annotator, aspell_speller, count_output)

        sentence = self._use_edits(edits, parsed_sentence)
        
        if use_token_level:
            sentence = self.introduce_token_level_errors_on_sentence(sentence.split(' '), aspell_speller, morfodita)

        if use_char_level:
            sentence = self.introduce_char_level_errors_on_sentence(sentence)

        return sentence
    
    def _use_edits(self, edits: List[Edit], parsed_sentence) -> str:
        if len(edits) == 0:
            return parsed_sentence.text
        docs = [edits[0].c_toks]
        prev_edit = edits[0]
        for edit in edits[1:]:
            next_docs = self._merge(prev_edit, edit, parsed_sentence)
            docs = next_docs + docs
            prev_edit = edit
        subtexts = [doc.text for doc in docs]

        sentence = parsed_sentence[:edits[-1].o_start].text + " " + " ".join(subtexts) + " " + parsed_sentence[edits[0].o_end:].text
        sentence = sentence.strip()
        return sentence

    def _merge(self, prev_edit: Edit, next_edit: Edit, parsed_sentence) -> List:
        if prev_edit.o_start > next_edit.o_end:
            docs = [next_edit.c_toks, parsed_sentence[next_edit.o_end:prev_edit.o_start]]
        else:
            docs = [next_edit.c_toks]
        return docs

def get_token_vocabulary():
    tokens = []
    with resources.open_text("src.retag.vocabularies", "vocabulary_cs.tsv") as reader:
        for line in reader:
            line = line.strip('\n')
            token, freq = line.split('\t')
            if token.isalpha():
                tokens.append(token)
    return tokens

def get_char_vocabulary(lang):
    if lang == 'cs':
        czech_chars_with_diacritics = 'áčďěéíňóšřťůúýž'
        czech_chars_with_diacritics_upper = czech_chars_with_diacritics.upper()
        allowed_chars = ', .'
        allowed_chars += string.ascii_lowercase + string.ascii_uppercase + czech_chars_with_diacritics + czech_chars_with_diacritics_upper
        return list(allowed_chars)


def main(args):
    char_vocabulary = get_char_vocabulary(args.lang)
    word_vocabulary = get_token_vocabulary()
    aspell_speller = aspell.Speller('lang', args.lang)
    morfodita = GenerateForms("src.utils.MorphoDiTa.czech-morfflex2.0-220710.dict")
    with open(args.error_config) as f:
        config = json.load(f)
    error_generator = ErrorGenerator(config, word_vocabulary, char_vocabulary,
                                     [0.2, 0.2, 0.2, 0.2, 0.2], 0.02, 0.01,
                                     [0.5, 0.2, 0.1, 0.05, 0.1, 0.05], 0.15, 0.2)
    error_generator._init_annotator()
    input_path = args.input
    output_path = args.output
    with open(input_path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()

            if args.format == "m2":
                error_sentence, m2_lines = error_generator.get_m2_edits_text(line, aspell_speller)
                with open(output_path, "a+") as output_file:
                    output_file.write("S " + error_sentence + "\n")
                    for m2_line in m2_lines:
                        output_file.write(m2_line + "\n")
                    output_file.write("\n")
            else:
                error_line = error_generator.create_error_sentence(
                    line, aspell_speller, False, False, morfodita, "counts.txt")
                with open(output_path, "a+") as output_file:
                    output_file.write(error_line + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Create m2 file with errors.")
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-f', '--format', type=str, default="m2")
    parser.add_argument('-o', '--output', type=str, default="output.m2")
    parser.add_argument('-l', '--lang', type=str)
    parser.add_argument('-e', '--error-config', type=str, default="defaul_errors.json")
    
    parser.add_argument('-c', '--count', action='store_true')
    return parser.parse_args()


def main_cli():
    args = parse_args()
    main(args)


if __name__ == "__main__":
    main_cli()