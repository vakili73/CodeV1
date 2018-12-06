from sklearn import metrics


class Reporter(object):
    def __init__(self, file_dir='./report.log'):
        self.file = open(file_dir, 'at')

    def end_line(self, end='\n'):
        self.file.write(end)

    def write_dataset(self, dataset):
        self.file.write('dataset,{},'.format(dataset))

    def write_schema(self, schema):
        self.file.write('schema,{},'.format(schema))

    def write_build(self, build):
        self.file.write('build,{},'.format(build))

    def write_way(self, way):
        self.file.write('way,{},'.format(way))

    def write_shot(self, shot):
        self.file.write('shot,{},'.format(shot))

    def write_augment(self, augment):
        self.file.write('augment,{},'.format(augment))

    def write_knn(self, weights, n_neighbors):
        self.file.write('{},{}-neighbors,'.format(weights, n_neighbors))

    def write_accuracy(self, score):
        self.file.write('accuracy,{},'.format(score*100))

    def write_f1_score(self, score):
        self.file.write('f1-score,{},'.format(score*100))

    def write_stack(self, dataset, schema, build, way, shot, augment):
        self.write_dataset(dataset)
        self.write_schema(schema)
        self.write_build(build)
        self.write_way(way)
        self.write_shot(shot)
        self.write_augment(augment)
        self.file.flush()

    def write_knn_metrics(self, weights, n_neighbors, accu_score, f1_score):
        self.write_knn(weights, n_neighbors)
        self.write_accuracy(accu_score)
        self.write_f1_score(f1_score)
        self.file.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
