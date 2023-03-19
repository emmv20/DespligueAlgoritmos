
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""A word-counting workflow."""

# pytype: skip-file

from __future__ import absolute_import


import argparse
import logging
import re
import os
import csv
import json
import random

from past.builtins import unicode

import apache_beam as beam
from apache_beam.io import ReadFromText #fichero de texto a colección
from apache_beam.io import WriteToText #colección a fichero de texto
from apache_beam.coders.coders import Coder #gestión de codificación de archivos
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions, DirectOptions

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

nltk.download("stopwords")

# CLEANING
STOP_WORDS = stopwords.words("english") #stopwords en inglés
STEMMER = SnowballStemmer("english") #stemmer en inglés
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+" #Expresión regular para limpiar urls


class CustomCoder(Coder): 
    """Custom coder utilizado para ller y escribir strings. Realiza una serie de tranformaciones entre codificaciones"""

    def __init__(self, encoding: str):
        # latin-1
        # iso-8859-1
        self.enconding = encoding

    def encode(self, value):
        return value.encode(self.enconding)

    def decode(self, value):
        return value.decode(self.enconding)

    def is_deterministic(self):
        return True

class ExtractColumnsDoFn(beam.DoFn):    #Extrae información de las columnas
    def process(self, element):
        data = json.loads(element)
        data = [data['content'], data['annotation']['label'][0]]
        #print(data)
        yield data

        
class PreprocessColumnsTrainFn(beam.DoFn):

    def process_sentiment(self, label):
        sentiment = int(label)
        if sentiment == 1:
            return "POSITIVE"
        else:
            return "NEGATIVE"

    def process_text(self, content):
        
        stem = True
        text = re.sub(TEXT_CLEANING_RE, " ", str(content).lower()).strip()
        tokens = []
        for token in text.split():
            if token not in STOP_WORDS:
                if stem:
                    tokens.append(STEMMER.stem(token))
                else:
                    tokens.append(token)
        return " ".join(tokens)

    def process(self, element):
        processed_text = self.process_text(element[0])
        processed_sentiment = self.process_sentiment(element[1]) 
        yield f"{processed_text}, {processed_sentiment}"







def run(argv=None, save_main_session=True):

  """Main entry point; defines and runs the wordcount pipeline."""



  parser = argparse.ArgumentParser()

  parser.add_argument(
        "--work-dir",
        dest="work_dir",
        required=True,
        help="Working directory",
  )

  parser.add_argument( # añado argumento. Fichero de entrada
        "--input",
        dest="input",
        required=True,
        help="Input dataset in work dir",
    )
  parser.add_argument( # añado argumento. Fichero de salida
        "--output",
        dest="output",
        required=True,
        help="Output path to store transformed data in work dir",
    )
  parser.add_argument( # añado argumento. Indica si estamos entrenando o evaluando
        "--mode",
        dest="mode",
        required=True,
        choices=["train", "test"],
        help="Type of output to store transformed data",
    )



  known_args, pipeline_args = parser.parse_known_args(argv)

  
  
  
  # Añadimos configuración de la pipeline
  pipeline_options = PipelineOptions(pipeline_args)
  pipeline_options.view_as(SetupOptions).save_main_session = save_main_session 
  pipeline_options.view_as(DirectOptions).direct_num_workers = 0

  
  
  
  
  # Hasta aquí hemos añadido la configuración, a partir de ahora comenzamos a construir la pipeline:
  with beam.Pipeline(options=pipeline_options) as p:
    
    
    
    raw_data = p | "ReadTwichData" >> ReadFromText(
            known_args.input,
            coder=CustomCoder("latin-1"))
    
    
    
    
    if known_args.mode == "train":

      transformed_data = (
                raw_data
                | "ExtractColumns" >> beam.ParDo(ExtractColumnsDoFn())
                | "Preprocess" >> beam.ParDo(PreprocessColumnsTrainFn()) 
            )
      #Vamos a construir un conjunto de validación
      eval_percent = 20
      assert 0 < eval_percent < 100, "eval_percent must in the range (0-100)"
      train_dataset, eval_dataset = (
                transformed_data
                | "Split dataset"
                >> beam.Partition( #particionamos nuestros datos cumpliendo
                    lambda elem, _: int(random.uniform(0, 100) < eval_percent), 2 
                ) 
            )

      train_dataset | "TrainWriteToCSV" >> WriteToText( # escribimos el conjunto de train en un csv
                os.path.join(known_args.output, "train", "part"), 
    file_name_suffix=".csv"
            )
      eval_dataset | "EvalWriteToCSV" >> WriteToText( # escribimos el conjunto de eval en un csv
                os.path.join(known_args.output, "eval", "part"), 
    file_name_suffix=".csv"
            )

    else:
      transformed_data = (
                raw_data
                | "ExtractColumns" >> beam.ParDo(ExtractColumnsDoFn())
                | "Preprocess" >> beam.Map(lambda x: f'"{x[0]}"')
            )

      transformed_data | "TestWriteToCSV" >> WriteToText(
                os.path.join(known_args.output, "test", "part"), 
    file_name_suffix=".csv"
            )


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    run()
