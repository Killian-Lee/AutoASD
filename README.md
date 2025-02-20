# AutoASD

## The code for AutoASD

The repo contains the source of the AutoASD model.



### Requirements

AutoASD uses the following dependencies with Python 3.8

- `pytorch==2.10`
- `networkx==2.8`
- `flask==3.1.0`
- `numpy==2.2.0`
- `gepandas==1.0.1`
- `matplotlib==3.8.0`



### Datasets

- Two folders are used in the system, ETH-anomaly and ETH-scalability.
- You can obtain the dataset from [1].



### Preprocessing

- You can refer [2] to preparing the datasets.
- Or you can use this scripts directly.

```
python preprocessing.py
```



### How to run the code for AutoASD

```
python app.py
```



### Reference

[1]https://drive.google.com/drive/folders/1NKjzJS7w1dDqvwMm-lQcP3ryf3X8AfGx

[2]https://github.com/tsourakakis-lab/antibenford-subgraphs/blob/main/eth_token_2018.ipynb
