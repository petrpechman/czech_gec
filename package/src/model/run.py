import argparse
import src.model.pipeline as pipeline
import src.model.evaluator as evaluator

import src.model.evaluator_predictions as eval_pred
import src.model.evaluator_files as eval_files
import src.model.evaluator_multi as eval_multi
import src.model.evaluator_predictions_optimizer as eval_pred_opt

def main(config_filename: str, eval: bool,
         pred: bool = False, files: bool = False, multi_eval: bool = False, opt: bool = False):

    if sum([multi_eval, pred, files, opt]) > 1:
        print("It is not possible to use more evaluator specific parameters together...")

    if eval:
        if multi_eval:
            dirs = config_filename.split(',')
            eval_multi.main(dirs)
        elif pred:
            eval_pred.main(config_filename)
        elif files:
            eval_files.main(config_filename)
        elif opt:
            eval_pred_opt.main(config_filename)
        else:
            evaluator.main(config_filename)
    else:
        pipeline.main(config_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, help="Config file.")
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction, help="Run evaluation.")
    parser.add_argument("--pred", action=argparse.BooleanOptionalAction, help="Create predictions.")
    parser.add_argument("--files", action=argparse.BooleanOptionalAction, help="Evaluate predicitions in files.")
    parser.add_argument("--multi_eval", action=argparse.BooleanOptionalAction, help="Evaluate more experiments.")
    parser.add_argument("--opt", action=argparse.BooleanOptionalAction, help="Eval with optimizer.")
    args = parser.parse_args()

    print(args.eval)
    main(args.config, args.eval, args.pred, args.files, args.multi_eval, args.opt)