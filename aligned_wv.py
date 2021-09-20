# sinamps
from fastText_multilingual.fasttext import FastVector
fa_dictionary = FastVector(vector_file='wiki.fa.align.vec')
en_dictionary = FastVector(vector_file='wiki.en.align.vec')
fa_vector = fa_dictionary["کتاب"]
en_vector = en_dictionary["book"]
print("Similarity between 'book' and 'کتاب' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["دفترچه"]
en_vector = en_dictionary["book"]
print("Similarity between 'book' and 'دفترچه' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["پیتزا"]
en_vector = en_dictionary["book"]
print("Similarity between 'book' and 'پیتزا' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["پیتزا"]
en_vector = en_dictionary["pizza"]
print("Similarity between 'pizza' and 'پیتزا' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["کوه"]
en_vector = en_dictionary["mountain"]
print("Similarity between 'mountain' and 'کوه' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["کوه"]
en_vector = en_dictionary["mount"]
print("Similarity between 'mount' and 'کوه' is ", FastVector.cosine_similarity(fa_vector, en_vector))
fa_vector = fa_dictionary["خوب"]
en_vector = en_dictionary["bad"]
print("Similarity between 'bad' and 'خوب' is ", FastVector.cosine_similarity(fa_vector, en_vector))
while(True):
    fa = input("Enter the word in Persian: ")
    en = input("Enter the word in English: ")
    fa_vector = fa_dictionary[fa]
    en_vector = en_dictionary[en]
    print("Similarity between '" + en + "' and '" + fa + "' is ", FastVector.cosine_similarity(fa_vector, en_vector))