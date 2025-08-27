A simple transformer architecture same as BERT made for understanding and training  of LLMS
The corpa used https://huggingface.co/datasets/jiajinda/bookcorpusopen

Trained the same as BERT, which uses Masked Language Modelling(MLM) and Next Sentence Prediction(NSP)

Note:
The NSP task gets very good results, maybe because my code shuffles the false sentences. This caused contamination between stories, which made it very easy for the model to predict the result. [This was an intentional design choice].
Finally, cleaning the corpa in a better way may give much better results.
