import numpy as np

# from edit import Edit
from .edit import Edit

from typing import List
from abc import ABC, abstractmethod
from errant.annotator import Annotator

from unidecode import unidecode

class Error(ABC):
    def __init__(self, target_prob: float, absolute_prob: float = 0.0, use_absolute_prob: bool = False) -> None:
        self.target_prob = target_prob
        self.absolute_prob = absolute_prob
        self.use_absolute_prob = use_absolute_prob
        self.num_errors = 0
        self.num_possible_edits = 0

    @abstractmethod
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        pass

    @staticmethod
    def _capitalize_words(variants: List) -> List:
        capital_variants = []
        for word, possible_errors, use_capital in variants:
            if use_capital:
                capital_possible_errors = []
                for possible_error in possible_errors:
                    possible_error.capitalize()
                capital_variants.append((word.capitalize(), capital_possible_errors))
        return capital_variants

    def _general(self, variants: List, parsed_sentence, annotator: Annotator, type_name: str) -> List[Edit]:
        first_word_variants = self._capitalize_words(variants)
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i == 0:
                for (right, possible_errors) in first_word_variants:
                    if token.text == right:
                        o_toks = annotator.parse(token.text)
                        for possible_error in possible_errors:
                            c_toks = annotator.parse(possible_error)
                            edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type=type_name)
                            edits.append(edit)
            for (right, possible_errors, _) in variants:
                if token.text == right:
                    o_toks = annotator.parse(token.text)
                    for possible_error in possible_errors:
                        c_toks = annotator.parse(possible_error)
                        edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type=type_name)
                        edits.append(edit)
        return edits
    
    @staticmethod
    def _general_try_retag(variants: List, edit: Edit, type_name: str) -> Edit:
        first_word_variants = Error._capitalize_words(variants)
        for (right, possible_errors) in first_word_variants:
            if edit.c_toks.text == right:
                for possible_error in possible_errors:
                    if edit.o_toks.text == possible_error:
                        edit.type = type_name
        for (right, possible_errors, _) in variants:
            if edit.c_toks.text == right:
                for possible_error in possible_errors:
                    if edit.o_toks.text == possible_error:
                        edit.type = type_name
        return edit
    
    @staticmethod
    # @abstractmethod
    def try_retag_edit(edit: Edit) -> Edit:
        pass


class ErrorMeMne(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == "mně":
                o_toks = annotator.parse("mně")
                c_toks = annotator.parse("mě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                edits.append(edit)
            elif token.text == "mě":
                o_toks = annotator.parse("mě")
                c_toks = annotator.parse("mně")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                edits.append(edit)
            if i == 0:
                if token.text == "Mně":
                    o_toks = annotator.parse("Mně")
                    c_toks = annotator.parse("Mě")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                    edits.append(edit)
                elif token.text == "Mě":
                    o_toks = annotator.parse("Mě")
                    c_toks = annotator.parse("Mně")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMne")
                    edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_toks.text == "mně" and edit.c_toks.text == "mě":
            edit.type = "MeMne"
        if edit.o_toks.text == "mě" and edit.c_toks.text == "mně":
            edit.type = "MeMne"
        if edit.o_toks.text == "Mě" and edit.c_toks.text == "Mně":
            edit.type = "MeMne"
        if edit.o_toks.text == "Mně" and edit.c_toks.text == "Mě":
            edit.type = "MeMne"
        return edit
    
class ErrorMeMneSuffix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and token.text.endswith("mně"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-3] + "mě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneSuffix")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.endswith("mě"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-2] + "mně")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneSuffix")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if len(edit.o_toks.text) > 3 and edit.o_toks.text.endswith("mně") and \
            len(edit.c_toks.text) > 2 and edit.c_toks.text.endswith("mě"):
            edit.type = "MeMneSuffix"
        if len(edit.o_toks.text) > 2 and edit.o_toks.text.endswith("mě") and \
            len(edit.c_toks.text) > 3 and edit.c_toks.text.endswith("mně"):
            edit.type = "MeMneSuffix"
        return edit

class ErrorMeMneIn(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and 'mně' in token.text and not token.text.endswith("mně"):
                index = token.text.find('mně')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "mě" + token.text[index+3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 2 and 'mě' in token.text and not token.text.endswith("mě"):
                index = token.text.find('mě')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "mně" + token.text[index+2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.startswith('Mě'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Mně" + token.text[2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
            elif len(token.text) > 3 and token.text.startswith('Mně'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Mě" + token.text[3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MeMneIn")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if len(edit.o_toks.text) > 3 and 'mně' in edit.o_toks.text and not edit.o_toks.text.endswith("mně") and \
            len(edit.c_toks.text) > 2 and 'mě' in edit.c_toks.text and not edit.c_toks.text.endswith("mě"):
            edit.type = "MeMneIn"
        if len(edit.o_toks.text) > 2 and 'mě' in edit.o_toks.text and not edit.o_toks.text.endswith("mě") and \
            len(edit.c_toks.text) > 3 and 'mně' in edit.c_toks.text and not edit.c_toks.text.endswith("mně"):
            edit.type = "MeMneIn"
        if len(edit.o_toks.text) > 3 and 'Mně' in edit.o_toks.text and not edit.o_toks.text.endswith("Mně") and \
            len(edit.c_toks.text) > 2 and 'Mě' in edit.c_toks.text and not edit.c_toks.text.endswith("Mě"):
            edit.type = "MeMneIn"
        if len(edit.o_toks.text) > 2 and 'Mě' in edit.o_toks.text and not edit.o_toks.text.endswith("Mě") and \
            len(edit.c_toks.text) > 3 and 'Mně' in edit.c_toks.text and not edit.c_toks.text.endswith("Mně"):
            edit.type = "MeMneIn"
        return edit
    
class ErrorDNT(Error):
    specific_chars = ['d', 't', 'n']

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        # specific_chars = ['d', 't', 'n']
        edits = []
        for i, token in enumerate(parsed_sentence):
            for specific_char in self.specific_chars:
                if specific_char + 'i' in token.text:
                    index = token.text.find(specific_char + 'i')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'y' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                elif specific_char + 'í' in token.text:
                    index = token.text.find(specific_char + 'í')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'ý' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                if specific_char + 'y' in token.text:
                    index = token.text.find(specific_char + 'y')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'i' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
                elif specific_char + 'ý' in token.text:
                    index = token.text.find(specific_char + 'ý')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'í' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="DNT")
                    edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        for specific_char in ErrorDNT.specific_chars:
            if specific_char + 'i' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'i')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'y' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "DNT"
            if specific_char + 'í' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'í')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'ý' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "DNT"
            if specific_char + 'y' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'y')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'i' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "DNT"
            if specific_char + 'ý' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'ý')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'í' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "DNT"
        return edit
    
class ErrorEnumeratedWord(Error):
    specific_chars = ['b', 'f', 'l', 'm', 'p', 's', 'v', 'z']

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        # specific_chars = ['b', 'f', 'l', 'm', 'p', 's', 'v', 'z']
        edits = []
        for i, token in enumerate(parsed_sentence):
            for specific_char in self.specific_chars:
                if specific_char + 'i' in token.text:
                    index = token.text.find(specific_char + 'i')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'y' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                elif specific_char + 'í' in token.text:
                    index = token.text.find(specific_char + 'í')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'ý' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                if specific_char + 'y' in token.text:
                    index = token.text.find(specific_char + 'y')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'i' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
                elif specific_char + 'ý' in token.text:
                    index = token.text.find(specific_char + 'ý')
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse(token.text[:index] + specific_char + 'í' + token.text[index+2:])
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="EnumeratedWord")
                    edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        for specific_char in ErrorEnumeratedWord.specific_chars:
            if specific_char + 'i' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'i')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'y' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "EnumeratedWord"
            if specific_char + 'í' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'í')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'ý' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "EnumeratedWord"
            if specific_char + 'y' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'y')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'i' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "EnumeratedWord"
            if specific_char + 'ý' in edit.o_toks.text:
                index = edit.o_toks.text.find(specific_char + 'ý')
                pos_c_toks = edit.o_toks.text[:index] + specific_char + 'í' + edit.o_toks.text[index+2:]
                if pos_c_toks == edit.c_toks.text:
                    edit.type = "EnumeratedWord"
        return edit

class ErrorSuffixIY(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.endswith("y") and len(token.text) > 1:
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "i")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("ý") and len(token.text) > 1:
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "í")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("i") and len(token.text) > 1:
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "y")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
            elif token.text.endswith("í") and len(token.text) > 1:
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-1] + "ý")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SuffixIY")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_toks.text.endswith('y') and len(edit.o_toks.text) > 1:
            pos_c_toks = edit.o_toks.text[:-1] + 'i'
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SuffixIY"
        if edit.o_toks.text.endswith('ý') and len(edit.o_toks.text) > 1:
            pos_c_toks = edit.o_toks.text[:-1] + 'í'
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SuffixIY"
        if edit.o_toks.text.endswith('i') and len(edit.o_toks.text) > 1:
            pos_c_toks = edit.o_toks.text[:-1] + 'y'
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SuffixIY"
        if edit.o_toks.text.endswith('í') and len(edit.o_toks.text) > 1:
            pos_c_toks = edit.o_toks.text[:-1] + 'ý'
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SuffixIY"
        return edit
    
class ErrorUU(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if 'ů' in token.text:
                index = token.text.find('ů')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "ú" + token.text[index+1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
            elif 'ú' in token.text:
                index = token.text.find('ú')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "ů" + token.text[index+1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
            elif token.text.startswith('Ú'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Ů" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="UU")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if 'ů' in edit.o_toks.text:
            index = edit.o_toks.text.find('ů')
            pos_c_toks = edit.o_toks.text[:index] + "ú" + edit.o_toks.text[index+1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "UU"
        if 'ú' in edit.o_toks.text:
            index = edit.o_toks.text.find('ú')
            pos_c_toks = edit.o_toks.text[:index] + "ů" + edit.o_toks.text[index+1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "UU"
        if edit.o_toks.text.startswith('Ů'):
            pos_c_toks = "Ú" + edit.o_toks.text[1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "UU"
        return edit
    

class ErrorCondional(Error):
    variants = [
            ("bychom", ["bysme", "by jsme"], True),
            ("byste", ["by ste", "by jste", "by jsi"], True),
            ("bych", ["by jsem", "bysem"], True),
            ("bys", ["by jsi", "by si"], True),
            ("abychom", ["abybysme", "aby jsme"], True),
            ("abyste", ["aby ste", "aby jste", "aby jsi"], True),
            ("abych", ["aby jsem", "abysem"], True),
            ("abys", ["aby jsi", "aby si"], True),
            ("kdybychom", ["kdybysme", "kdyby jsme"], True),
            ("kdybyste", ["kdyby ste", "kdyby jste", "kdyby jsi"], True),
            ("kdybych", ["kdyby jsem", "kdybysem"], True),
            ("kdybys", ["kdyby jsi", "kdyby si"], True),
        ]
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        # self.variants = [
        #     ("bychom", ["bysme", "by jsme"], True),
        #     ("byste", ["by ste", "by jste", "by jsi"], True),
        #     ("bych", ["by jsem", "bysem"], True),
        #     ("bys", ["by jsi", "by si"], True),
        #     ("abychom", ["abybysme", "aby jsme"], True),
        #     ("abyste", ["aby ste", "aby jste", "aby jsi"], True),
        #     ("abych", ["aby jsem", "abysem"], True),
        #     ("abys", ["aby jsi", "aby si"], True),
        #     ("kdybychom", ["kdybysme", "kdyby jsme"], True),
        #     ("kdybyste", ["kdyby ste", "kdyby jste", "kdyby jsi"], True),
        #     ("kdybych", ["kdyby jsem", "kdybysem"], True),
        #     ("kdybys", ["kdyby jsi", "kdyby si"], True),
        # ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "Conditional")
    
    def try_retag_edit(edit: Edit) -> Edit:
        edit = Error._general_try_retag(ErrorCondional.variants, edit, "Conditional")
        return edit
    
class ErrorSpecificWords(Error):
    variants = [
            ("viz", ["viz."], False),
            ("výjimka", ["vyjímka"], True),
            ("seshora", ["zeshora", "zezhora"], True),
        ]
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        # self.variants = [
        #     ("viz", ["viz."], False),
        #     ("výjimka", ["vyjímka"], True),
        #     ("seshora", ["zeshora", "zezhora"], True),
        # ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "SpecificWords")
    
    def try_retag_edit(edit: Edit) -> Edit:
        edit = Error._general_try_retag(ErrorSpecificWords.variants, edit, "SpecificWords")
        return edit
    
class ErrorSZPrefix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 1 and token.text.startswith("s") and token.text != "se":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("z" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("z") and token.text != "ze":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("s" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("S") and token.text != "Se":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Z" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
            elif len(token.text) > 1 and token.text.startswith("Z") and token.text != "Ze":
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("S" + token.text[1:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="SZPrefix")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if len(edit.o_toks.text) > 1 and edit.o_toks.text.startswith("s") and edit.o_toks.text != "se":
            pos_c_toks = "z" + edit.o_toks.text[1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SZPrefix"
        if len(edit.o_toks.text) > 1 and edit.o_toks.text.startswith("z") and edit.o_toks.text != "ze":
            pos_c_toks = "s" + edit.o_toks.text[1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SZPrefix"
        if len(edit.o_toks.text) > 1 and edit.o_toks.text.startswith("S") and edit.o_toks.text != "Se":
            pos_c_toks = "Z" + edit.o_toks.text[1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SZPrefix"
        if len(edit.o_toks.text) > 1 and edit.o_toks.text.startswith("Z") and edit.o_toks.text != "Ze":
            pos_c_toks = "S" + edit.o_toks.text[1:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "SZPrefix"
        return edit

class ErrorNumerals(Error):
    variants = [
            ("oběma", ["oběmi, oboumi", "obouma"], True),
            ("dvěma", ["dvěmi", "dvouma"], True),
            ("třemi", ["třema"], True),
            ("čtyřmi", ["čtyřma"], True),
        ]
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        # self.variants = [
        #     ("oběma", ["oběmi, oboumi", "obouma"], True),
        #     ("dvěma", ["dvěmi", "dvouma"], True),
        #     ("třemi", ["třema"], True),
        #     ("čtyřmi", ["čtyřma"], True),
        # ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "Numerals")
    
    def try_retag_edit(edit: Edit) -> Edit:
        edit = Error._general_try_retag(ErrorNumerals.variants, edit, "Numerals")
        return edit
    
class ErrorMyMi(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == "mi":
                o_toks = annotator.parse("mi")
                c_toks = annotator.parse("my")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                edits.append(edit)
            elif token.text == "my":
                o_toks = annotator.parse("my")
                c_toks = annotator.parse("mi")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                edits.append(edit)
            if i == 0:
                if token.text == "Mi":
                    o_toks = annotator.parse("Mi")
                    c_toks = annotator.parse("My")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                    edits.append(edit)
                elif token.text == "My":
                    o_toks = annotator.parse("My")
                    c_toks = annotator.parse("Mi")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="MyMi")
                    edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_toks.text == "mi" and edit.c_toks.text == "my":
            edit.type = "MyMi"
        if edit.o_toks.text == "my" and edit.c_toks.text == "mi":
            edit.type = "MyMi"
        if edit.o_toks.text == "Mi" and edit.c_toks.text == "My":
            edit.type = "MyMi"
        if edit.o_toks.text == "My" and edit.c_toks.text == "Mi":
            edit.type = "MyMi"
        return edit
    
class ErrorBeBjeSuffix(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and token.text.endswith("bje"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-3] + "bě")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeSuffix")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.endswith("bě"):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:-2] + "bje")
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeSuffix")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if len(edit.o_toks.text) > 3 and edit.o_toks.text.endswith("bje"):
            pos_c_toks = edit.o_toks.text[:-3] + "bě"
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeSuffix"
        if len(edit.o_toks.text) > 2 and edit.o_toks.text.endswith("bě"):
            pos_c_toks = edit.o_toks.text[:-2] + "bje"
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeSuffix"
        return edit
    
class ErrorBeBjeIn(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if len(token.text) > 3 and 'bje' in token.text and not token.text.endswith("bje"):
                index = token.text.find('bje')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "bě" + token.text[index+3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 2 and 'bě' in token.text and not token.text.endswith("bě"):
                index = token.text.find('bě')
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text[:index] + "bje" + token.text[index+2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 2 and token.text.startswith('Bě'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Bje" + token.text[2:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
            elif len(token.text) > 3 and token.text.startswith('Bje'):
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse("Bě" + token.text[3:])
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="BeBjeIn")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if len(edit.o_toks.text) > 3 and 'bje' in edit.o_toks.text and not edit.o_toks.text.endswith("bje"):
            index = edit.o_toks.text.find('bje')
            pos_c_toks = edit.o_toks.text[:index] + "bě" + edit.o_toks.text[index+3:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeIn"
        if len(edit.o_toks.text) > 2 and 'bě' in edit.o_toks.text and not edit.o_toks.text.endswith("bě"):
            index = edit.o_toks.text.find('bě')
            pos_c_toks = edit.o_toks.text[:index] + "bje" + edit.o_toks.text[index+2:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeIn"
        if len(edit.o_toks.text) > 3 and edit.o_toks.text.startswith("Bje"):
            index = edit.o_toks.text.find('Bje')
            pos_c_toks = "Bě" + edit.o_toks.text[3:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeIn"
        if len(edit.o_toks.text) > 2 and 'Bě' and edit.o_toks.text.startswith("Bě"):
            index = edit.o_toks.text.find('Bě')
            pos_c_toks = "Bje" + edit.o_toks.text[2:]
            if pos_c_toks == edit.c_toks.text:
                edit.type = "BeBjeIn"
        return edit
    
class ErrorSebou(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        prev_token = None
        for i, token in enumerate(parsed_sentence):
            if 'sebou' == token.text:
                if prev_token and prev_token == "s":
                    o_toks = annotator.parse("s sebou")
                    c_toks = annotator.parse("sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
                else:
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("s sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
            if i == 0:
                if 'Sebou' == token.text:
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("S sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
            if i == 1:
                if prev_token == 'S' and token.text == 'sebou':
                    o_toks = annotator.parse(token.text)
                    c_toks = annotator.parse("Sebou")
                    edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="Sebou")
                    edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_toks.text == "sebou" and edit.c_toks.text == "s sebou":
            edit.type = "Sebou"
        if edit.o_toks.text == "s sebou" and edit.c_toks.text == "sebou":
            edit.type = "Sebou"
        if edit.o_toks.text == "Sebou" and edit.c_toks.text == "S sebou":
            edit.type = "Sebou"
        if edit.o_toks.text == "S sebou" and edit.c_toks.text == "Sebou":
            edit.type = "Sebou"
        return edit

class ErrorSentenceFirstUpper(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            token = parsed_sentence[0]
            if token.text.istitle():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.lower())
                edit = Edit(o_toks, c_toks, [0, len(o_toks), 0, len(c_toks)], type="SentenceFirstUpper")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == 0 and edit.o_end == 1 and edit.o_toks.text.islower() and edit.o_toks.text.title() == edit.c_toks.text:
            edit.type = "SentenceFirstUpper"
        return edit

class ErrorSentenceFirstLower(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            token = parsed_sentence[0]
            if token.text.islower():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.title())
                edit = Edit(o_toks, c_toks, [0, len(o_toks), 0, len(c_toks)], type="SentenceFirstLower")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == 0 and edit.o_end == 1 and edit.o_toks.text.istitle() and edit.o_toks.text.lower() == edit.c_toks.text:
            edit.type = "SentenceFirstLower"
        return edit
    
class ErrorTitleToLower(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i > 0 and token.text.istitle():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.lower())
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="TitleToLower")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start != 0 and edit.o_end != 1 and edit.o_toks.text.islower() and edit.o_toks.text.title() == edit.c_toks.text:
            edit.type = "TitleToLower"
        return edit
    
class ErrorLowerToTitle(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if i > 0 and token.text.islower():
                o_toks = annotator.parse(token.text)
                c_toks = annotator.parse(token.text.title())
                edit = Edit(o_toks, c_toks, [i, i+len(o_toks), i, i+len(c_toks)], type="LowerToTitle")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start != 0 and edit.o_end != 1 and edit.o_toks.text.istitle() and edit.o_toks.text.lower() == edit.c_toks.text:
            edit.type = "LowerToTitle"
        return edit

class ErrorPrepositionSZ(Error):
    variants = [
            ("s", ["se"], True),
            ("z", ["ze"], True),
            ("se", ["s"], True),
            ("ze", ["z"], True),
        ]
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        # self.variants = [
        #     ("s", ["se"], True),
        #     ("z", ["ze"], True),
        #     ("se", ["s"], True),
        #     ("ze", ["z"], True),
        # ]
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        return self._general(self.variants, parsed_sentence, annotator, "PrepositionSZ")
    
    def try_retag_edit(edit: Edit) -> Edit:
        edit = Error._general_try_retag(ErrorPrepositionSZ.variants, edit, "PrepositionSZ")
        return edit

class ErrorCommaAdd(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.forbidden_neighbors = [',', '.']

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        prev_token = None
        for i, token in enumerate(parsed_sentence):
            if i > 0:
                if prev_token.text not in self.forbidden_neighbors and token.text not in self.forbidden_neighbors:
                    o_toks = annotator.parse("")
                    c_toks = annotator.parse(',')
                    edit = Edit(o_toks, c_toks, [i, i, i, i+1], type="CommaAdd")
                    edits.append(edit)
            prev_token = token
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == edit.o_end + 1 and edit.o_toks.text == ',' and edit.c_toks.text == "":
            edit.type = "CommaAdd"
        return edit

class ErrorCommaRemove(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text == ',':
                o_toks = annotator.parse(",")
                c_toks = annotator.parse('')
                edit = Edit(o_toks, c_toks, [i, i+1, i, i], type="CommaRemove")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == edit.o_end and edit.c_toks.text == ',':
            edit.type = "CommaRemove"
        return edit
    
class ErrorRemoveDiacritics(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            word = token.text
            if unidecode(word) != word:
                o_toks = annotator.parse(word)
                c_toks = annotator.parse(unidecode(word))
                edit = Edit(o_toks, c_toks, [i, i+1, i, i+1], type="RemoveDiacritics")
                edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == edit.c_start and \
           edit.o_end == edit.c_end and \
           edit.o_toks.text == unidecode(edit.c_toks.text):
            edit.type = "RemoveDiacritics"
        return edit
    
class ErrorAddDiacritics(Error):
    def __init__(self, target_prob: float, absolute_prob: float = 0, use_absolute_prob: bool = False) -> None:
        super().__init__(target_prob, absolute_prob, use_absolute_prob)
        self.czech_diacritics_dict = {
            'a': ['á'], 'c': ['č'], 'd': ['ď'], 'e': ['é', 'ě'],'i': ['í'], 'n': ['ň'], 
            'o': ['ó'],'r': ['ř'], 's': ['š'],'t': ['ť'], 'u': ['ů', 'ú'], 'y': ['ý'], 'z': ['ž']}
        self.czech_diacritizables_chars = [k for k in self.czech_diacritics_dict.keys()] + \
            [k.upper() for k in self.czech_diacritics_dict.keys()]

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            word = token.text
            indices = []
            count = 0
            for j, c in enumerate(word):
                if c in self.czech_diacritizables_chars:
                    indices.append(j)
                    count += 1
            
            if len(indices) == 0:
                continue

            numbers_of_changes = np.arange(count + 1)
            probs = np.float_power(10, -numbers_of_changes)
            probs = probs / np.sum(probs)

            number_of_changes = np.random.choice(numbers_of_changes, 1, replace=True, p=probs)
            if len(number_of_changes) == 0:
                continue
            indices_to_change = np.random.choice(indices, size=number_of_changes, replace=False)

            new_word = []
            for j, char in enumerate(word):
                if j in indices_to_change:
                    changed_char = np.random.choice(self.czech_diacritics_dict[char.lower()])
                    if char.isupper():
                        changed_char = changed_char.upper()
                    char = changed_char
                new_word.append(char)
            new_word = "".join(new_word)

            o_toks = annotator.parse(word)
            c_toks = annotator.parse(new_word)
            edit = Edit(o_toks, c_toks, [i, i+1, i, i+1], type="AddDiacritics")
            edits.append(edit)
        return edits
    
    def try_retag_edit(edit: Edit) -> Edit:
        if edit.o_start == edit.c_start and \
           edit.o_end == edit.c_end and \
           edit.o_toks.text != unidecode(edit.c_toks.text) and \
           unidecode(edit.o_toks.text) == unidecode(edit.c_toks.text):
            edit.type = "AddDiacritics"
        return edit


ERRORS = {
    "MeMne": ErrorMeMne,
    "MeMneSuffix": ErrorMeMneSuffix,
    "MeMneIn": ErrorMeMneIn,
    "SuffixIY": ErrorSuffixIY,
    "DTN": ErrorDNT,
    "EnumeratedWord": ErrorEnumeratedWord,
    "UU": ErrorUU,
    "Conditional": ErrorCondional,
    "SpecificWords": ErrorSpecificWords,
    "SZPrefix": ErrorSZPrefix,
    "Numerals": ErrorNumerals,
    "MyMi": ErrorMyMi,
    "BeBjeSuffix": ErrorBeBjeSuffix,
    "BeBjeIn": ErrorBeBjeIn,
    "Sebou": ErrorSebou,
    "SentenceFirstUpper": ErrorSentenceFirstUpper,
    "SentenceFirstLower": ErrorSentenceFirstLower,
    "TitleToLower": ErrorTitleToLower,
    "LowerToTitle": ErrorLowerToTitle,
    "PrepositionSZ": ErrorPrepositionSZ,
    "CommaAdd": ErrorCommaAdd,
    "CommaRemove": ErrorCommaRemove,
    "RemoveDiacritics": ErrorRemoveDiacritics,
    "AddDiacritics": ErrorAddDiacritics,
}

### NO USED:

class ErrorReplace(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.isalpha():
                proposals = aspell_speller.suggest(token.text)[:10]
                if len(proposals) > 0:
                    new_token_text = np.random.choice(proposals)
                    c_toks = annotator.parse(new_token_text)
                    edit = Edit(token, c_toks, [i, i+1, i, i+1], type="Replace")
                    edits.append(edit)
        return edits


class ErrorInsert(Error):
    def __init__(self, target_prob: float, word_vocabulary) -> None:
        super().__init__(target_prob)
        self.word_vocabulary = word_vocabulary

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            new_token_text = np.random.choice(self.word_vocabulary)
            c_toks = annotator.parse(new_token_text)
            edit = Edit(token, c_toks, [i, i, i, i+1], type="Insert")
            edits.append(edit)
        return edits


class ErrorDelete(Error):
    def __init__(self, target_prob: float) -> None:
        super().__init__(target_prob)
        self.allowed_source_delete_tokens = [',', '.', '!', '?']

    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.isalpha() and token.text not in self.allowed_source_delete_tokens:
                c_toks = annotator.parse("")
                edit = Edit(token, c_toks, [i, i+1, i, i], type="Remove")
                edits.append(edit)
        return edits


class ErrorRecase(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        for i, token in enumerate(parsed_sentence):
            if token.text.islower():
                new_token_text = token.text[0].upper() + token.text[1:]
            else:
                num_recase = min(len(token.text), max(1, int(np.round(np.random.normal(0.3, 0.4) * len(token.text)))))
                char_ids_to_recase = np.random.choice(len(token.text), num_recase, replace=False)
                new_token_text = ''
                for char_i, char in enumerate(token.text):
                    if char_i in char_ids_to_recase:
                        if char.isupper():
                            new_token_text += char.lower()
                        else:
                            new_token_text += char.upper()
                    else:
                        new_token_text += char
            c_toks = annotator.parse(new_token_text)
            edit = Edit(token, c_toks, [i, i+1, i, i+1], type="Recase")
            edits.append(edit)
        return edits


class ErrorSwap(Error):
    def __call__(self, parsed_sentence, annotator: Annotator, aspell_speller = None) -> List[Edit]:
        edits = []
        if len(parsed_sentence) > 1:
            previous_token = parsed_sentence[0]
            for i, token in enumerate(parsed_sentence[1:]):
                i = i + 1
                c_toks = annotator.parse(token.text + " " + previous_token.text)
                edit = Edit(token, c_toks, [i-1, i+1, i-1, i+1], type="Swap")
                edits.append(edit)
                previous_token = token
        return edits

###