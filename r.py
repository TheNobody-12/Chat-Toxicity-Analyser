# convert csv  to text in specific format
import pandas as pd
data = pd.read_csv('toxicity-main\\toxicity-main\\toxicity_en.csv')

#  prepand tex: to each row
data['text'] = 'text: ' + data['text'].astype(str)
# convert to text file
# df = data['text']

data['text'].to_csv('text.txt', sep='\n', index=False, header=False)