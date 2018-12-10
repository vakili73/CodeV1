

class Reporter(object):
    def __init__(self, file_dir='./report.log'):
        self.file = open(file_dir, 'at')

    def end_line(self, end='\n'):
        self.file.write(end)

    def write_dataset(self, dataset):
        self.file.write('dataset,{},'.format(dataset))
        return self

    def write_schema(self, schema):
        self.file.write('schema,{},'.format(schema))
        return self

    def write_build(self, build):
        self.file.write('build,{},'.format(build))
        return self

    def write_way(self, way):
        self.file.write('way,{},'.format(way))
        return self

    def write_shot(self, shot):
        self.file.write('shot,{},'.format(shot))
        return self

    def write_augment(self, augment):
        self.file.write('augment,{},'.format(augment))
        return self

    def write_knn(self, weights, n_neighbors):
        self.file.write('{},{}-neighbors,'.format(weights, n_neighbors))
        return self

    def write_accuracy(self, score):
        self.file.write('accuracy,{},'.format(score*100))
        return self

    def write_f1_score(self, score):
        self.file.write('f1-score,{},'.format(score*100))
        return self

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
