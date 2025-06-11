import csv
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import os

def language_eval_bleu(model_recover_path, eval_model, preds):
    # Create a results folder
    results_dir = '/kaggle/working/results'
    os.makedirs(results_dir, exist_ok=True)

    # Generate paths
    filename_base = os.path.splitext(os.path.basename(model_recover_path))[0]
    filename_base = os.path.join(results_dir, f"{filename_base}{eval_model}")
    reference_path = f"{filename_base}_gt.csv"
    hypothesis_path = f"{filename_base}.csv"

    list_of_list_of_references = []
    list_of_list_of_hypotheses = []

    with open(reference_path, 'w', newline='') as gt, open(hypothesis_path, 'w', newline='') as gen:
        gt_writer = csv.writer(gt)
        gen_writer = csv.writer(gen)

        for preds_dict in tqdm(preds, total=len(preds)):
            reference = preds_dict.get('gt_caption', "").strip()
            candidate = preds_dict.get('gen_caption', "").strip()

            gt_writer.writerow([reference])
            gen_writer.writerow([candidate])

            ref_tokens = reference.split()
            cand_tokens = candidate.split()

            list_of_list_of_references.append([ref_tokens])  # [[ref1, ref2, ...]]
            list_of_list_of_hypotheses.append(cand_tokens)

    # Compute BLEU scores
    bleu_1gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(1, 0, 0, 0))
    bleu_2gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4gram = corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

    # Print results
    print(f'1-Gram BLEU: {bleu_1gram:.4f}')
    print(f'2-Gram BLEU: {bleu_2gram:.4f}')
    print(f'3-Gram BLEU: {bleu_3gram:.4f}')
    print(f'4-Gram BLEU: {bleu_4gram:.4f}')

    return bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram
