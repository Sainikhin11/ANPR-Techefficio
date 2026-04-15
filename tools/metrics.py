from Levenshtein import distance


def compute_cer(pred, gt):
    if len(gt) == 0:
        return 0
    return distance(pred, gt) / len(gt)


class MetricsTracker:
    def __init__(self):
        self.total = 0
        self.correct = 0
        self.cer_sum = 0.0
        self.plates_detected = 0
        self.plates_read = 0

    def update(self, pred, gt):
        self.total += 1
        if pred == gt:
            self.correct += 1
        self.cer_sum += compute_cer(pred, gt)

    def add_counts(self, detected=0, read=0):
        self.plates_detected += detected
        self.plates_read += read

    def report(self):
        acc = self.correct / self.total if self.total else 0
        cer = self.cer_sum / self.total if self.total else 0
        return acc, cer
