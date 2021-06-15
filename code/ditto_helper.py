import re
import json
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd

def removeprefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    else:
        return text

def to_ditto_format(df_input, output_file):
  file = open(output_file,'w')
  left_atts = [col for col in df_input.columns if col.startswith('left')]
  right_atts = [col for col in df_input.columns if col.startswith('right')]
  df_dict = df_input.to_dict(orient='records')
  for row in df_dict:
    content = ''
    for attr in left_atts:
      content += 'COL %s VAL %s ' % (removeprefix(attr,'left_'), re.sub('\t', ' ', str(row[attr])))
    content += ' \t '
    for attr in right_atts:
      content += 'COL %s VAL %s ' % (removeprefix(attr,'right_'), re.sub('\t', ' ', str(row[attr])))
    content += ' \t%s\n' % (row['label'])
    file.write(content)
  file.close()

def to_jsonl(input_file, output_file):
  input_dict = pd.read_csv(input_file).to_dict('records')
  with open(output_file, 'w') as outfile:
    for row in input_dict:
      left_dict = {}
      right_dict = {}
      for k,v in row.items():
        if k.startswith('left'):
            left_dict[removeprefix(k,'left_')]=v
        if k.startswith('right'):
            right_dict[removeprefix(k,'right_')]=v
      row_list = [left_dict,right_dict]
      json.dump(row_list, outfile)
      outfile.write('\n')