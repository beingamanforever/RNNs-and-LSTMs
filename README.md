# RNNs-and-LSTMs
Usage of RNNs and LSTMs architecture for multivarate as well as univariate modelling of panel data and time series data. It helps in predicting the future trends, by analysing window sections of the historical data.
# About Work-2 
Here I analyzed the energy consumption dataset from Kaggle (previously present in UCI), and analysed multiple variables which affect Application Electricity consumption, and used them to predict the future values of energy consumption. Thus the target value is application energy consumption and the input variables would be application energy consumption (historical data), lights, TH_1, RH_1, Visibility. 

> [!NOTE]
> The UCI repository is not working, so u may refer to kaggle to access the updated dataset (I have uploaded the kaggle version of the dataset in my repo)

All the columns of the dataset me be seen here

![im1](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/4cd78a95-5e27-4a6d-89e7-7d416e064b6e)

'Date' column entries had class as object was changed to datetime, for plotting purposes

![image](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/80a5934a-686e-4d37-81ae-4bc666e8675d)

> [!NOTE]
> Do follow the above method for converting to datetime using pandas, as it's much more efficient than parsing than it's counterpart (refer to the image above)

I then plotted the relationship between various variables which were used as variables and analysed their spread, this helped me undertsand if normalization is required.

![RNN LSTM-1](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/3950e66f-82b7-4b95-986f-7cd3df511c96)

Then I used MinMaxScaling() to normalize the values, as it would help in converging the gradients faster and also the outliers would be removed. That's why I preferred it over StandardScaler as it removed the outliers in 'Application' column, I came to an conclusion that it has outliers as the 75% quartile was at 100ish value and the max was at 1080, also when I checked the number of rows of values about the 75% it counted to a significant number, thus indicating the presence of outliers.

Then after more pre-processing and cleaning of data and spliting the data into train & test datasets, I modelled a LSTM architechture which was trained using my training dataset, validated using validation_dataset and accuracy was then predicted using test_dataset.

> [!NOTE]
> Do make sure to keep shuffle=False, as orelse the ordering of data maybe changed and hence the soul agenda behind the time series sequences would be lost.
> 
![image](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/0f4f940d-3b42-4a6f-8bc8-da74119776c2)

Then, I used regularisation techniques to improve the accuracy like adding droupout layers to avoid overfitting, since I don't have GPU in my system hence I then transferred to Google Collab to use the T4_GPU provided as it reduces the training time by a major factor. I also used early stopping so as to reduce the training time, which allowed the algorithm to not run through all epochs. 


![image](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/c51a6642-aab5-449d-87b6-605e3a0343d9)

After that I compiled my model and I used optimizer as 'adam' , loss as 'mse' and metrics as MeanAbsoluteError(). Then I saved the results obtained from each epoch into a variable named as history, which would be then used for plotting purposes.

![image](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/51a7b8c9-643a-4e25-9d2c-d60c6047b06c)


Finally I plotted the actual and predicted values of the target variable to analyse how closely they resemble each other

![RNN LSTM-2](https://github.com/beingamanforever/RNNs-and-LSTMs/assets/121532863/42b07b17-9c81-4608-b5b3-7efd6e8d7c68)

# Where to go from here
> [!IMPORTANT]
> Further improvements I am considering to make on this model
1. Hyper parameter tuning of the parameters of LSTM, and the training variables like epochs, batch_size can be optimized.
2. Changing the LSTM architecture, trying out various different models and analyzing the best from it (changes like adding more layers, tunning the droupout parameter, using different activation functions like 'relu' or 'tanh', instead of Leaky ReLU. As predominantly, RNN based models use these as activation functions in their hidden state.
3. Using different scaling methods and analysing their performance.
4. Varing the length of the sliding window that was used for predicting the future prices.
5. Parsing and caching of data to improve speed, thereby reducing the training time.

> [!TIP]
> If u want to run the code, consider running it on google collab and adjusting the runtime parameter to T4_GPU, instead of CPU if u don't have a fast machine locally. Orelse, u may simply run it on local system.
