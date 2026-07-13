from nltk.tokenize import word_tokenize, sent_tokenize
from pymystem3 import Mystem
import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

class TextProcessor:
    """
        A pipeline that maps written text (lemmas) to sign language video clips.
        Uses exact dictionary lookup first, then falls back to FAISS similarity
        search on semantic embeddings if a word is out-of-vocabulary.
        """

    def __init__(self, csv_path, embeddings_path, gloss_words_path, video_folder):
        # 1. Initialize the Russian morphological lemmatizer.
        # This loads a heavy dictionary into memory. We store it on 'self'
        # so we don't have to re-initialize it for every sentence.
        self.m = Mystem()

        # Store the root folder where the actual .mp4 video files are stored.
        self.video_folder = video_folder

        # 2. Load the gloss-to-video mapping from a TSV (tab-separated) file.
        # The CSV is expected to have at least two columns: 'gloss_norm' and 'cut_name'.
        self.df = pd.read_csv(csv_path, sep='\t')

        # Defensive programming: cast the gloss column to strings.
        # Without this, if a gloss is numeric (e.g., "123"), Pandas might read it
        # as an integer, causing a KeyError when used as a dictionary key later.
        self.df['gloss_norm'] = self.df['gloss_norm'].astype(str)

        # Build a dictionary: gloss_word -> list_of_video_filenames.
        # We use a list because a single sign might have multiple video variations
        # (e.g., different speeds, angles, or signers).
        # Example: {'EAT': ['eat1.mp4', 'eat2.mp4'], 'DRINK': ['drink.mp4']}
        self.gloss_map = self.df.groupby('gloss_norm')['cut_name'].apply(list).to_dict()

        # 3. Load the pre-computed numerical embeddings (dense vectors) for all glosses.
        # This is a 2D NumPy array of shape (num_glosses, embedding_dimension).
        # Each row corresponds to the vector representation of a specific gloss word.
        self.loaded_gloss_embeddings = np.load(embeddings_path)

        # Load the list of gloss words that correspond to the rows in the embeddings file.
        # CRITICAL ASSUMPTION: The order of words in this file MUST exactly match
        # the row order in 'embeddings_path'. If they are misaligned, FAISS will
        # return the wrong gloss for a given index.
        with open(gloss_words_path, 'r', encoding='utf-8') as f:
            self.saved_gloss_words = [line.strip() for line in f]

        # 4. Build the FAISS index for ultra-fast nearest-neighbor search.
        # Extract the dimension of the embedding vectors (e.g., 512).
        dim = self.loaded_gloss_embeddings.shape[1]

        # 'IndexFlatL2' uses brute-force exact search with L2 (Euclidean) distance.
        # It is exact (not approximate) and highly optimized in C++.
        # Note: FAISS does NOT copy the data automatically; we must explicitly add it.
        self.index = faiss.IndexFlatL2(dim)

        # Add the entire embedding matrix to the index.
        # The data is copied into FAISS's internal C++ memory structure.
        self.index.add(self.loaded_gloss_embeddings)

        # 5. Load the SentenceTransformer model for the semantic fallback mechanism.
        # This model is used to generate embeddings for unseen words ON THE FLY.
        # It is multilingual, which is crucial if the input text is in Russian.
        self.model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')

    def lemmatize_sentence(self, sentence):
        """
        Lemmatize a sentence and return a list of clean, alphabetic lemmas.
        Example: "Я читаю книги" -> ['я', 'читать', 'книга']
        """
        # Tokenize the raw string into individual tokens (words and punctuation).
        # Example: "Hello, world!" -> ['Hello', ',', 'world', '!']
        words = word_tokenize(sentence)

        # Rejoin the tokens into a single string for Mystem, then lemmatize.
        # Mystem returns a list of lemmas (one per token), but often includes
        # empty strings, newlines, and punctuation markers.
        lemmatized = [
            lemma
            for lemma in self.m.lemmatize(" ".join(words))
            # Filter out non-words:
            # 1. 'lemma.strip()' removes empty strings or whitespace-only entries.
            # 2. 'lemma.isalpha()' ensures we only keep actual alphabetic characters
            #    (A-Z, a-z, Cyrillic). This drops numbers, punctuation, and special
            #    Mystem control characters like '{' or '|'.
            if lemma.strip() and lemma.isalpha()
        ]

        return lemmatized


    def get_videos_from_text(self, input_text):
        """
        Process the input text and return a list of video file paths.
        For each lemma in the text, returns a tuple:
            (matched_gloss_norm, full_video_path)
        If no suitable gloss is found for a word, returns (None, None).
        The order of the output list matches the chronological order of words.
        """
        # 1. Split the input text into sentences based on punctuation (. ! ?).
        # This is done before lemmatization to allow for modular processing,
        # and it helps avoid memory issues when processing extremely long texts.
        sentences = sent_tokenize(input_text)

        # 2. Lemmatize each sentence individually.
        # This results in a list of lists: e.g., [['я', 'читать'], ['книга']]
        all_lemmatized_sentences = [self.lemmatize_sentence(s) for s in sentences]

        # 3. Flatten the list of lists into a single 1D list.
        # The nested comprehension preserves the original word order.
        # e.g., ['я', 'читать', 'книга']
        all_lemmatized_words = [word for sentence in all_lemmatized_sentences for word in sentence]

        # 4. Process each lemma sequentially to find the corresponding video.
        results = []
        for lemma in all_lemmatized_words:
            # Initialize placeholders for this iteration.
            matched_gloss_norm = None
            video_path_full = None

            # --- TIER 1: EXACT DICTIONARY MATCH (Fast Path) ---
            # O(1) dictionary lookup. This handles ~80-90% of common words.
            if lemma in self.gloss_map:
                matched_gloss_norm = lemma

                # Select the first video file associated with this gloss.
                # If there are multiple variations, this heuristic picks the default.
                video_filename = self.gloss_map[lemma][0]

                # Construct the full file system path using the OS-appropriate
                # separator (e.g., '/' on Linux, '\' on Windows).
                video_path_full = os.path.join(self.video_folder, video_filename)

            # --- TIER 2: SEMANTIC FALLBACK (FAISS + ML) ---
            # This handles Out-Of-Vocabulary (OOV) words.
            else:
                # Generate a dense vector embedding for the unseen lemma.
                # We pass it as a list (batch of 1) and convert directly to NumPy.
                query_embedding = self.model.encode([lemma], convert_to_numpy=True)

                # Query the FAISS index to find the single nearest neighbor.
                # D = array of L2 distances, I = array of row indices.
                D, I = self.index.search(query_embedding, 1)

                # Extract the integer index of the closest known gloss.
                neighbor_idx = I[0][0]

                # Look up the actual gloss string from our stored list.
                similar_gloss = self.saved_gloss_words[neighbor_idx]

                # Double-check that the retrieved gloss actually exists in the map.
                # (Defensive programming in case of data corruption or mismatch.)
                if similar_gloss in self.gloss_map:
                    matched_gloss_norm = similar_gloss

                    # Again, take the first available video file for this gloss.
                    video_filename = self.gloss_map[similar_gloss][0]
                    video_path_full = os.path.join(self.video_folder, video_filename)
                else:
                    # Fallback failed completely for this word.
                    matched_gloss_norm = None
                    video_path_full = None

            # Append the result tuple for this specific lemma.
            results.append(video_path_full)

        # Return the list of results. The order here directly corresponds
        # to the order of words in 'all_lemmatized_words'.
        return results



