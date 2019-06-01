import  pandas as pd
from pandas.io.json import json_normalize
import json


def format_json_journeys():

    json1_file = open('exampleJSONforEDA.json')
    json1_str = json1_file.read()
    json1_data = json.loads(json1_str)

    final_json = []

    
    for register in json1_data:
        for key, value in register.items():
            for key2,value2 in value.items():
                for key3, value3 in value2.items():
                    if(key3!="opportunities" and key3 != "client"):
                        for key4, value4 in value3.items():
                            final_json.append(value4)
                            for key5, value5 in value4.items():
                                print(key2)
                                print(key3)
                    else:
                        continue
    return final_json



final_json = format_json_journeys()
with open('data.json', 'w') as outfile:
    json.dump(final_json, outfile, ensure_ascii=False)


