import fasttext
import fasttext.util
from gensim.models import KeyedVectors
from tqdm import tqdm
import multiprocessing as mp
import numpy as np

def process_chunk(chunk):
    return {word: ft.get_word_vector(word) for word in chunk}

def parallelize_conversion(ft_model, words, num_processes):
    # Split words into chunks
    chunk_size = len(words) // num_processes
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]

    # Create a pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        # Map the process_chunk function to the chunks
        results = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing chunks"))

    # Combine results
    word_vectors = {}
    for result in results:
        word_vectors.update(result)

    return word_vectors

if __name__ == "__main__":
    # Load the fastText model
    ft = fasttext.load_model('cc.ar.300.bin')

    # Get all words
    words = ft.get_words()

    # Determine the number of processes to use (you can adjust this)
    num_processes = mp.cpu_count()

    # Parallelize the conversion
    word_vectors = parallelize_conversion(ft, words, num_processes)

    # Convert to gensim format
    gensim_model = KeyedVectors(ft.get_dimension())
    gensim_model.add_vectors(list(word_vectors.keys()), list(word_vectors.values()))

    # Save in Word2Vec format
    gensim_model.save_word2vec_format('arabic_word2vec_model.bin', binary=True)