import editdistance

def calc_wer(target_text: str, pred_text: str):
    splitted_target_text = target_text.split(' ')
    if len(splitted_target_text) == 0:
        return 1
    return editdistance.distance(splitted_target_text, pred_text.split(' ')) / len(splitted_target_text)
    

def calc_cer(target_text: str, pred_text: str):
    if len(target_text) == 0:
        return 1
    return editdistance.distance(target_text, pred_text) / len(target_text)
