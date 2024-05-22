import argparse
import pipeline
import evaluator
import inference
import inference_opt
import evaluate_files
import evaluator_multi


def main(config_filename: str, eval: bool, infer: bool = False, eval_preds: bool = False, multi_eval: bool = False, infer_opt: bool = False):
    if sum([multi_eval, infer, eval_preds, infer_opt]) > 1:
        print('It is not possible to combine these parameters...')
        return

    if eval:
        if multi_eval:
            dirs = config_filename.split(',')
            evaluator_multi.main(dirs)
        elif infer:
            inference.main(config_filename)
        elif infer_opt:
            inference_opt.main(config_filename)
        elif eval_preds:
            evaluate_files.main(config_filename)
        else:
            evaluator.main(config_filename)
    else:
        pipeline.main(config_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run complete evaluation if not specified.")
    parser.add_argument("--infer", action=argparse.BooleanOptionalAction, help="Create inferences and save them into files.")
    parser.add_argument("--eval_preds", action=argparse.BooleanOptionalAction, help="Evaluate predictions saved in files.")
    parser.add_argument("--multi_eval", action=argparse.BooleanOptionalAction, help="Evaluate over more experiments.")
    parser.add_argument("--infer_opt", action=argparse.BooleanOptionalAction, help="Create inferences and save them into files. Use checkpoint with optimizer.")
    args = parser.parse_args()

    main(args.config, args.eval, args.infer, args.eval_preds, args.multi_eval, args.infer_opt)