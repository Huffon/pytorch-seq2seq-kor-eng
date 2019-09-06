## Sequence-to-sequence PyTorch implementations
This repo contains various sequential models used to **translate Korean sentence into English sentence**.

I used translation dataset, but you can apply these models to any sequence to sequence, text generation tasks such as text summarization, response generation, ...

All of base codes are based on this great [**seq2seq tutorial**](https://github.com/bentrevett/pytorch-seq2seq).

In this project, I specially used Korean-English translation corpus from [**AI Hub**](http://www.aihub.or.kr/) to apply torchtext into Korean dataset. 

I can upload the dataset because it requires a Approval from AI Hub. You can can an approval from AI Hub, if you request it to the administrator.

And I also used [**soynlp**](https://github.com/lovit/soynlp) library which is used to tokenize Korean sentence. 
It is really nice and easy to use, you should try :)

<br/>

### Overview
- Number of train data: 86,200
- Number of validation data: 36,900
- Number of test data: 41,000
```
Example: 
{
{'kor': '['이', '문서는', '제출', '할', '필요', '없어요']', 
 'eng': '['You', 'do', "n't", 'need', 'to', 'submit', 'this', 'document', '.']'}
}
```
<br/>

### Requirements

- Following libraries are fundamental to this repo. Since I used conda environment `requirements.txt` has much more dependent libraries. 
- Therefore if you encounters any dependency problem, just use this command `pip install -r requirements.txt`

```
en-core-web-sm==2.1.0
numpy==1.16.4
pandas==0.25.1
scikit-learn==0.21.3
soynlp==0.0.493
spacy==2.1.8
torch==1.0.1
torchtext==0.4.0
```
<br/>

### Usage
- Before training the model, you should train soynlp tokenizer and build vocabulary using following code. 
- You can determine the size of vocabulary of Korean and English dataset. In general, Korean sentences creates larger size vocabulary. Therefore to make balance between korean and english dataset, you have to pick proper vocab size
- By running this code, you will get `tokenizer.pickle`, `kor.pickle` and `eng.pickle` which are used to train, 
test model and predict user's input sentence

```
python build_pickle.py --kor_vocab KOREAN_VOCAB_SIZE --eng_vocab ENGLISH_VOCAB_SIZE
```


- For training, run `main.py` with train mode (default option)

```
python main.py --model MODEL_NAME --save_model MODEL_NAME_TO_SAVE
```

- For testing, run `main.py` with test mode

```
python main.py --model MODEL_NAME --mode test --save_model SAVED_MODEL
```

- For predicting, run `predict.py` with your input sentence. 
- *Don't forget to wrap your input sentence with double quotation mark !*

```
python predict.py --model MODEL_NAME --input "YOUR_KOREAN_INPUT" --save_model SAVED_MODEL
```