import pickle

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith('numpy._core'):
            m = m.replace('numpy._core', 'numpy.core')
        return super().find_class(m, n)

v = RenameUnpickler(open('models/nlp/tfidf_vectorizer.pkl', 'rb')).load()
print('Type:', type(v))
print('Attributes:', dir(v))
print('Has idf_:', hasattr(v, 'idf_'))
if hasattr(v, 'idf_'):
    print('idf_ shape:', v.idf_.shape)
