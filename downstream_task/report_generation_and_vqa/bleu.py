import csv
import os
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

def language_eval_bleu(model_recover_path, eval_model, preds):
    """
    Compute BLEU scores with smoothing.
    Writes predicted and ground truth captions to CSV files.
    """
    smoother = SmoothingFunction().method1

    # Create results directory
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)

    # Construct output file paths
    filename_base = os.path.splitext(os.path.basename(model_recover_path))[0]
    filename_base = os.path.join(results_dir, filename_base + str(eval_model))
    reference_path = filename_base + '_gt.csv'
    hypothesis_path = filename_base + '.csv'

    # Data containers
    list_of_list_of_references = []
    list_of_list_of_hypotheses = []

    # Write CSVs and prepare BLEU input
    with open(reference_path, 'w', newline='') as gt, open(hypothesis_path, 'w', newline='') as gen:
        gt_writer = csv.writer(gt)
        gen_writer = csv.writer(gen)

        for preds_dict in tqdm(preds, total=len(preds)):
            reference = preds_dict.get('gt_caption', '')
            candidate = preds_dict.get('gen_caption', '')

            gt_writer.writerow([str(reference)])
            gen_writer.writerow([str(candidate)])

            ref_tokens = reference.split()
            cand_tokens = candidate.split()

            list_of_list_of_references.append([ref_tokens])  # Each reference wrapped in a list
            list_of_list_of_hypotheses.append(cand_tokens)

    # BLEU scores with smoothing
    bleu_1gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses,
                             weights=(1, 0, 0, 0), smoothing_function=smoother)
    bleu_2gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses,
                             weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    bleu_3gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses,
                             weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    bleu_4gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses,
                             weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    # Print results
    print(f'1-Gram BLEU: {bleu_1gram:.2f}')
    print(f'2-Gram BLEU: {bleu_2gram:.2f}')
    print(f'3-Gram BLEU: {bleu_3gram:.2f}')
    print(f'4-Gram BLEU: {bleu_4gram:.2f}')

    return bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram
