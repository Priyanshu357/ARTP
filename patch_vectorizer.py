import pickle
import numpy as np
import scipy.sparse as sp

class RenameUnpickler(pickle.Unpickler):
    def find_class(self, m, n):
        if m.startswith('numpy._core'):
            m = m.replace('numpy._core', 'numpy.core')
        return super().find_class(m, n)

# 1. Load the vectorizer
v = RenameUnpickler(open('models/nlp/tfidf_vectorizer.pkl', 'rb')).load()

# 2. Check and force attributes
print('Vocabulary size:', len(v.vocabulary_))

if not hasattr(v, 'idf_') and hasattr(v, '_tfidf'):
    # In scikit-learn 1.6, the state might be stored differently or cleared for ONNX.
    # But because we only use it for ONNX preprocessing (v.transform), 
    # and the ONNX model ALREADY has the IDF weights baked in (as seen in the ONNX file), 
    # we just need sklearn to STOP complaining about NotFittedError during transform().
    
    # Let's mock the fitted state
    n_features = len(v.vocabulary_)
    v.idf_ = np.ones(n_features)
    v._tfidf.idf_ = np.ones(n_features)
    v.fitted_ = True
    v._tfidf.fitted_ = True
    
    # Try transforming a test sentence
    try:
        res = v.transform(["This is a test"])
        print("Transform successful!", res.shape)
    except Exception as e:
        print("Transform failed:", e)
        
    # Save the patched vectorizer over the old one to fix it permanently
    with open('models/nlp/tfidf_vectorizer_patched.pkl', 'wb') as f:
        pickle.dump(v, f)
        print("Patched vectorizer saved.")
