import csv
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import nltk

def language_eval_bleu(model_recover_path, eval_model, preds):
    # reference_path = model_recover_path.split('.')[0] + str(eval_model) + '_gt.csv'
    # hypothesis_path = model_recover_path.split('.')[0] + str(eval_model) + '.csv'
    import os

    # Create /kaggle/working/results directory if it doesn't exist
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)

    # Extract base name (e.g., "pytorch_model3054") without extension
    filename_base = os.path.splitext(os.path.basename(model_recover_path))[0]
    filename_base = os.path.join(results_dir, filename_base + str(eval_model))

    # Create paths
    reference_path = filename_base + '_gt.csv'
    hypothesis_path = filename_base + '.csv'


    list_of_list_of_references = []
    list_of_list_of_hypotheses = []

    with open(reference_path, 'w', newline='') as gt, open(hypothesis_path, 'w', newline='') as gen:
        gt_writer = csv.writer(gt)
        gen_writer = csv.writer(gen)

        for preds_dict in tqdm(preds, total=len(preds)):
            reference, candidate = "", ""
            for key, value in preds_dict.items():
                if key == 'gt_caption':
                    reference = value
                elif key == 'gen_caption':
                    candidate = value

            gt_writer.writerow([str(reference)])
            gen_writer.writerow([str(candidate)])

            ref_tokens = reference.split()
            cand_tokens = candidate.split()

            list_of_list_of_references.append([ref_tokens])  # list of one list of ref
            list_of_list_of_hypotheses.append(cand_tokens)

    bleu_1gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(1, 0, 0, 0))
    bleu_2gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    print(f'1-Gram BLEU: {bleu_1gram:.2f}')
    print(f'2-Gram BLEU: {bleu_2gram:.2f}')
    print(f'3-Gram BLEU: {bleu_3gram:.2f}')
    print(f'4-Gram BLEU: {bleu_4gram:.2f}')

    return bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram
