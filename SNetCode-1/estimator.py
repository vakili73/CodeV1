

def compile_fit(algorithm, transformed_db):
    optimizer = algorithm.optimizer()
    algorithm.model.compile(optimizer=optimizer,
                            loss=algorithm.loss,
                            metrics=algorithm.metric)
    algorithm.model.summary()
    module = 'algorithm.' + algorithm.name
    module = __import__(module)
    module = getattr(module, algorithm.name)
    history = getattr(module, "fit")(algorithm, transformed_db)
    return history
