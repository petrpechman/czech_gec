import errant
import argparse

# from edit import Edit
# from create_errors import ErrorGenerator
# from errors import ERRORS

from .edit import Edit
from .create_errors import ErrorGenerator
from .errors import ERRORS

lang = 'cs'
annotator = errant.load(lang)

def simplify_edits(sentence, edits: list[str], selected_coder: int = 0) -> list[Edit]:
    out_edits = []
    for edit in edits:
        # Preprocessing
        edit = edit[2:].split("|||") # Ignore "A " then split.
        span = edit[0].split()
        start = int(span[0])
        end = int(span[1])
        cat = edit[1]
        cor = edit[2]
        coder = int(edit[-1])
        if coder == selected_coder:
            o_toks = sentence[start:end]
            c_toks = annotator.parse(cor)
            edit_indices = [start, end, start, start + len(c_toks)]
            out_edit = Edit(o_toks, c_toks, edit_indices, cat, selected_coder)
            out_edits.append(out_edit)
    return out_edits

def retag_edits(line_edits: list[str]) -> list[Edit]:
    sentence = annotator.parse(line_edits[0][2:])
    source_edits = line_edits[1:]
    all_edits = []
    for selected_coder in range(5):
        edits = simplify_edits(sentence, source_edits, selected_coder)
        edits = ErrorGenerator.sort_edits(edits, True)

        for edit in edits:
            for error_class  in ERRORS.values():
                edit = error_class.try_retag_edit(edit)
            if not edit.type in ERRORS.keys():
                edit.type = "OTHER"

        edits = ErrorGenerator.sort_edits(edits, False)
        all_edits = all_edits + edits
    return all_edits
    

def main(args):
    input_filepath = args.input
    output_filepath = args.output

    counts = dict()
    for name in ERRORS.keys():
        counts[name] = 0
    counts['OTHER'] = 0

    line_edits = []
    with open(input_filepath, 'r') as file, open(output_filepath, 'w') as out_file:
        for line in file:
            line = line.strip()
            if line:
                line_edits.append(line)
            elif len(line_edits) > 0 :
                edits = retag_edits(line_edits)
                for edit in edits:
                    counts[edit.type] += 1
                m2_edits = [edit.to_m2() for edit in edits]
                
                out_file.write(line_edits[0] + '\n')
                for m2_line in m2_edits:
                    out_file.write(m2_line + '\n')
                out_file.write('\n')
                line_edits = []

        if len(line_edits) > 0 :
            edits = retag_edits(line_edits)
            for edit in edits:
                counts[edit.type] += 1
            m2_edits = [edit.to_m2() for edit in edits]
            
            out_file.write(line_edits[0] + '\n')
            for m2_line in m2_edits:
                out_file.write(m2_line + '\n')
            out_file.write('\n')
            line_edits = []
    
    if args.count:
        with open("counts.txt", 'w') as f:  
            for key, value in counts.items():  
                f.write('%s:%s\n' % (key, value))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create m2 file with errors.")
    parser.add_argument('-i', '--input', type=str)
    parser.add_argument('-o', '--output', type=str, default='out.m2')
    parser.add_argument('-c', '--count', action='store_true')

    args = parser.parse_args()
    main(args)