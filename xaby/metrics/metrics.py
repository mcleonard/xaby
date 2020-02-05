def accuracy(predictions, targets):
    metric = (predictions.data.argmax(axis=1) == targets.data).mean()
    return metric
