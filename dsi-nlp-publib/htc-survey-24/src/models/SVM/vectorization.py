from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.SVM.preprocessing import process_list_dataset


def tfidf_processing(train_x, test_x, config):
    print("Starting preprocessing ...")
    # This preprocessing does tokenization and text cleanup as in deeptriage
    # Simple filtering is done in read functions
    x_train, x_test = process_list_dataset(train_x, test_x,
                                           remove_garbage=config["remove_garbage"],
                                           stop_words_removal=config["stop_words_removal"])
    print("Preprocessing complete.\n")
    # Create feature vectors
    vectorizer = TfidfVectorizer(max_features=config["MAX_FEATURES"], ngram_range=(1, 2))
    # Train the feature vectors
    train_vectors = vectorizer.fit_transform(x_train)
    test_vectors = vectorizer.transform(x_test)
    return train_vectors, test_vectors
