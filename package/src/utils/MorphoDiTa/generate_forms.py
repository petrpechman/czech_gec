#!/usr/bin/env python3
import ufal.morphodita


class GenerateForms:
    def __init__(self, path: str = "./czech-morfflex2.0-220710.dict") -> None:
        self._morpho = ufal.morphodita.Morpho.load(path)

    def forms(self, form: str, derinet_distance: int = 0) -> set[str]:
        lemmas, tagged_lemmas = set(), ufal.morphodita.TaggedLemmas()
        if self._morpho.analyze(form, self._morpho.NO_GUESSER, tagged_lemmas) < 0:
            return lemmas
        for tagged_lemma in tagged_lemmas:
            lemma = tagged_lemma.lemma
            lemmas.add(lemma)

            if derinet_distance:
                derivator = self._morpho.getDerivator()
                currents = set([lemma])
                derivated_parent, derivated_children = ufal.morphodita.DerivatedLemma(), ufal.morphodita.DerivatedLemmas()
                for _ in range(derinet_distance):
                    neighbors = set()
                    for current in currents:
                        if derivator.parent(current, derivated_parent):
                            neighbors.add(derivated_parent.lemma)
                        if derivator.children(current, derivated_children):
                            neighbors.update(child.lemma for child in derivated_children)
                    original_lemmas_len = len(lemmas)
                    lemmas |= neighbors
                    if len(lemmas) == original_lemmas_len:
                        break
                    currents = neighbors

        forms, tagged_forms = set(), ufal.morphodita.TaggedLemmasForms()
        for lemma in lemmas:
            if self._morpho.generate(lemma, "", self._morpho.NO_GUESSER, tagged_forms) < 0:
                continue
            for tagged_lemma_forms in tagged_forms:
                for form in tagged_lemma_forms.forms:
                    forms.add(form.form)

        return forms


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--derinet_distance", default=0, type=int, help="Derinet distance")
    args = parser.parse_args()

    generator = GenerateForms()

    for line in sys.stdin:
        line = line.rstrip("\r\n")
        print(generator.forms(line, args.derinet_distance))
