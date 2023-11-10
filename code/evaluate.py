# pip install --user -U nltk
# https://www.nltk.org/install.html
import nltk
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

'''
Please check the above link how BLEU works:
https://www.nltk.org/_modules/nltk/translate/bleu_score.html
'''

bleu_weights = { "bleu-1": [1.0], "bleu-2": [0.5, 0.5], "bleu-3": [0.333, 0.333, 0.333], "bleu-4": [0.25, 0.25, 0.25, 0.25]}


def evaluate_dataset_new(model, data_loader, vocab, device):
    model.eval()

    # print(vocab.stoi['.']);exit()
    point_stoi =  vocab.stoi['.']
    references_total = []
    predictions_total = []
    for i in range(1):
        for idx, (imgs, captions) in tqdm(enumerate(data_loader), total=len(data_loader)):
            reference_batch = []
            prediction_batch = []
            for j in range(captions.size(1)):
                reference = []

                for i in range(1, captions.size(0)):
                    sentence_id = captions[i,j].item()
                    if sentence_id == point_stoi or sentence_id == 2: break
                    reference.append(vocab.itos[sentence_id])
                # print(reference)
                reference_batch.append(reference)
            references_total.append(reference_batch)    
            
            imgs = imgs.to(device)
            captions = captions.to(device)
            
            outputs = model(imgs, captions[:-1])
            # print('\n\n')
            outputs = outputs.argmax(2)
            # print()
            for j in range(outputs.size(1)):
                prediction = []
                for i in range(1, outputs.size(0)):
                    sentence_id = outputs[i,j].item()
                    if sentence_id ==point_stoi or sentence_id == 2: break
                    prediction.append(vocab.itos[sentence_id])
                # print(prediction)
                predictions_total.append(prediction)
                break

    bleu_1 = corpus_bleu(references_total, predictions_total, weights=bleu_weights["bleu-1"]) * 100

    model.train()
    return bleu_1
