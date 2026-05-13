import pickle

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith('numpy._core'):
            m = m.replace('numpy._core', 'numpy.core')
        return super().find_class(m, n)

v = RenameUnpickler(open('models/nlp/tfidf_vectorizer.pkl', 'rb')).load()
print('Has _tfidf:', hasattr(v, '_tfidf'))
if hasattr(v, '_tfidf'):
    print('Has _tfidf.idf_:', hasattr(v._tfidf, 'idf_'))
    if hasattr(v._tfidf, 'idf_'):
        v.idf_ = v._tfidf.idf_
        print('Patch successful, hasattr idf_ is now:', hasattr(v, 'idf_'))
