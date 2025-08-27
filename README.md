A simple transformer architecture same as BERT made for understanding and training  of LLMS
The corpa used https://huggingface.co/datasets/jiajinda/bookcorpusopen

Trained the same as BERT, which uses Masked Language Modelling(MLM) and Next Sentence Prediction(NSP)

Note:
The NSP task gets very good results, maybe because my code shuffles the false sentences. This caused contamination between stories, which made it very easy for the model to predict the result. [This was an intentional design choice].
And secondly, during Corpa cleaning, the very small, fewer than 10 character paragraphs were removed mainly because of the chapter headlines. Then that was converted into strings. This may result in NON CONTIGUOUS PARAGRAPHS TO END UP IN THE NSP TRUE SEGMENTS
Finally, cleaning the corpa in a better way may give much better results.


Running Steps:

#Cleaning + Split
1. CorpaClining.ipynb
2. To_Sent.ipynb
3. NSPSpleet.py

#Tokenisation
4. NSPSegmentTorch.py

#Training
5. TrainBERT-Main.ipynb
