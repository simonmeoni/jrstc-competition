def layerwise_learning_rate_decay(
    model,
    learning_rate,
    weight_decay=0.01,
    layerwise_learning_rate_factor=0.7,
    layerwise_learning_rate_flag=True,
):
    if layerwise_learning_rate_flag:
        no_decay = ["bias", "LayerNorm.weight"]
        # initialize lr for task specific layer
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.fc.named_parameters()],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
        ]
        # initialize lrs for every layer
        layers = [model.model.embeddings] + list(model.model.encoder.layer)
        layers.reverse()
        lr = learning_rate
        optimizer_grouped_parameters = []
        for layer in layers:
            lr *= layerwise_learning_rate_factor
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": weight_decay,
                    "lr": lr,
                },
                {
                    "params": [
                        p
                        for n, p in layer.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                    "lr": lr,
                },
            ]
        return optimizer_grouped_parameters
    else:
        return model.parameters()
