

class Report(object):
    def __init__(self, file_dir='./report.log'):
        self.file = open(file_dir, 'at')

    def end_line(self, end='\n'):
        self.file.write(end)
        self.flush()

    def write_text(self, txt):
        self.file.write('{},'.format(txt))
        return self

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
        self.file.write('weights,{},neighbors,{},'.format(weights, n_neighbors))
        return self

    def write_svm(self, kernel):
        self.file.write('kernel,{},'.format(kernel))
        return self

    def write_top_k_accuracy(self, score, k):
        self.file.write('top_{}_accu,{},'.format(k ,score*100))
        return self

    def write_accuracy(self, score):
        self.file.write('accuracy,{},'.format(score*100))
        return self

    def write_f1score(self, score):
        self.file.write('f1score,{},'.format(score*100))
        return self

    def write_precision(self, score):
        self.file.write('precision,{},'.format(score*100))
        return self

    def write_recall(self, score):
        self.file.write('recall,{},'.format(score*100))
        return self

    def write_precision_recall_f1score(self, precision, recall, f1score):
        self.write_precision(precision)
        self.write_recall(recall)
        self.write_f1score(f1score)
        self.flush()

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()
