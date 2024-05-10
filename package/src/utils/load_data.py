import aspell
import random

from typing import List
from typing import Optional
from multiprocessing import Pool
from multiprocessing import Queue

import src.utils.introduce_errors as introduce_errors
import src.utils.create_errors as create_errors
import src.utils.MorphoDiTa.generate_forms as GenerateForms

class GenereteErrorLine():
    '''
    Creates synthetic mistakes by aspell_speller and scripts from introduce_errors. 
    '''

    def __init__(self, tokens, characters, lang, token_err_distribution, char_err_distribution, token_err_prob, char_err_prob, token_std_dev=0.2, char_std_dev=0.01):
        self.tokens = tokens
        self.characters = characters
        self.lang = lang
        self.token_err_distribution = token_err_distribution
        self.char_err_distribution = char_err_distribution
        self.token_err_prob = token_err_prob
        self.token_std_dev = token_std_dev
        self.char_err_prob = char_err_prob
        self.char_std_dev = char_std_dev

    def __call__(self, line, aspell_speller):
        token_replace_prob, token_insert_prob, token_delete_prob, token_swap_prob, recase_prob = self.token_err_distribution
        char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob, change_diacritics_prob = self.char_err_distribution
        line = line.strip('\n')
        
        # introduce word-level errors
        line = introduce_errors.introduce_token_level_errors_on_sentence(line.split(' '), token_replace_prob, token_insert_prob, token_delete_prob,
                                                        token_swap_prob, recase_prob, float(self.token_err_prob), float(self.token_std_dev),
                                                        self.tokens, aspell_speller)
        if '\t' in line or '\n' in line:
            raise ValueError('!!! Error !!! ' + line)
        # introduce spelling errors
        line = introduce_errors.introduce_char_level_errors_on_sentence(line, char_replace_prob, char_insert_prob, char_delete_prob, char_swap_prob,
                                                       change_diacritics_prob, float(self.char_err_prob), float(self.char_std_dev),
                                                       self.characters)
        return line
    

def data_loader(filename, queue, start_position, end_position, gel: GenereteErrorLine, tokenizer, max_length, errors_from_file: bool,
                reverted_pipeline: bool, error_generator: create_errors.ErrorGenerator, lang: str, count_output: Optional[str]):
    # Starts read from start to end position, line with mistake is created for every read line,
    # then these lines are tokenized and store into dict that is putted into queue.
    counter = 0
    if not errors_from_file:
        aspell_speller = aspell.Speller('lang', lang)
        morfodita = GenerateForms("../utils/MorphoDiTa/czech-morfflex2.0-220710.dict")
    
    if error_generator:
        error_generator._init_annotator()

    with open(filename, 'r') as f:
        # find start position
        while counter != start_position:
            f.readline()
            counter += 1

        # read until end position
        while counter != end_position:
            line = f.readline()
            if len(line) > 0:
                line = line[:-1] if line[-1] == "\n" else line
            try:
                if errors_from_file:
                    line, error_line = line.split('\t', 1)
                else:
                    if error_generator is not None:
                        error_line = error_generator.create_error_sentence(
                            line.strip(), aspell_speller, True, True, morfodita, count_output)
                    else:
                        error_line = gel(line, aspell_speller)
                    

                if reverted_pipeline:
                    error_line, line = line, error_line

                tokenized = tokenizer(error_line, text_target=line, max_length=max_length, truncation=True, return_tensors="np")
            
                input_ids = tokenized['input_ids'][0]
                attention_mask = tokenized['attention_mask'][0]
                tokenized_target_line = tokenized['labels'][0]

                dato = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "tokenized_target_line": tokenized_target_line,
                    "original_sentence": error_line,
                    "correct_sentence": line,
                }
            
                queue.put(dato)
            except Exception as e:
                print(e)
                print(f"skip line: {line}")

            counter += 1

            if not line: # EOF
                f.seek(0) 
                counter = 0


def process_file_in_chunks(
        queue: Queue, pool: Pool, num_parallel: int, filename: str, file_size: int, 
        gel: GenereteErrorLine, tokenizer, max_length, errors_from_file: bool, reverted_pipeline: bool,
        error_generator: create_errors.ErrorGenerator, lang: str, count_output: Optional[str]):
    # Computes start index and end index for every process, stores them as arguments,
    # runs these processes and wait until they finished.
    
    start = random.randint(0, file_size-1)
    process_size = file_size // num_parallel

    # create list of arguments
    arguments = []

    current = start
    start_position = current
    for i in range(num_parallel):
        current = (current + process_size) % file_size
        end_position = current
        arguments.append((filename, queue, start_position, end_position, gel, 
                          tokenizer, max_length, errors_from_file, reverted_pipeline, 
                          error_generator, lang, count_output, ))
        start_position = current
    end_position = start
    arguments.append((filename, queue, start_position, end_position, gel, 
                      tokenizer, max_length, errors_from_file, reverted_pipeline, 
                      error_generator, lang, count_output, ))

    # start processes and wait until they finished
    pool.starmap(data_loader, arguments)


def data_generator(queue: Queue, files: List[str], num_parallel: int, gel: GenereteErrorLine, tokenizer, max_length, errors_from_file: bool = False,
                   reverted_pipeline: bool = False, error_generator: create_errors.ErrorGenerator = None, lang: str = "cs", count_output: Optional[str] = None):
    # Main methon that is used in pipeline.py
    # Creates pools and goes iteratively over files (one or more files).
    # Computes file size and run process_file_in_chunks. 
    index = 0
    pool = Pool(num_parallel)

    while True:
        file = files[index]

        # get file size
        with open(file, 'r') as f:
            for count, _ in enumerate(f):
                pass
        file_size = count + 1
        
        process_file_in_chunks(
            queue, pool, num_parallel, file, file_size, gel, tokenizer, max_length, 
            errors_from_file, reverted_pipeline, error_generator, lang, count_output)

        index += 1
        if index == len(files):
            index = 0
