# A list of notebooks to experiment with neural models related to finance

[S4 Model](./annotated-s4.ipynb) based on [The Annotated S4 ny srush](https://srush.github.io/annotated-s4/)

2024-01-10

I've forked @srush's repo and tried predicting hourly bitcoin price.

```python
git clone https://github.com/stillonearth/annotated-s4-finance

cd annotated-s4-finance
# pip install -r requirements-gpu.txt
python -m s4.train dataset=crypto layer=s4 train.epochs=100 model.d_model=256 model.layer.N=64
```

Metrics

```
=>> Epoch 100 Metrics ===
	Train Loss: 1.93076 -- Train Accuracy: 0.5448
	 Test Loss: 0.63157 --  Test Accuracy: 0.8281
	Best Test Loss: 0.26783 -- Best Test Accuracy: 0.9375 at Epoch 85
```