from difflib import SequenceMatcher

def exact_match_reward(pred, gt):

    return float(
        pred.strip().lower()
        ==
        gt.strip().lower()
    )

def similarity_reward(pred, gt):

    return SequenceMatcher(
        None,
        pred.lower(),
        gt.lower()
    ).ratio()

def length_penalty(pred):

    return -0.001 * len(pred.split())

def compute_reward(pred, gt):

    reward = 0.0

    reward += 0.5 * exact_match_reward(pred, gt)
    reward += 0.5 * similarity_reward(pred, gt)

    reward += length_penalty(pred)

    return reward
