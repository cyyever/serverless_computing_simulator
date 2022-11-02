# serverless_computing_simulator

A simulator for serverless computing

## dataset

This folder contains code for parsing and analyzing the traces of Microsoft's Azure Functions offered by Azure team on the [github repository](https://github.com/Azure/AzurePublicDataset). AzureFunctionsDataset2019.md provides the dataset URL and describes the data format.

_azure_trace.py_ is used to parse and cache the traces. _azure_analysis.py_ is used to plot the distributions of different parts of traces.

To invoke the scripts, you need to set the azure traces path in _conf/conf/azure.yaml_, then

```
cd simulation
python3 dataset/azure_analysis.py --config-name azure
```
