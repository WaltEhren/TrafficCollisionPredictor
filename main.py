# Daten verarbeiten:
# Folge dem tutorial.
# monat tag jahr splitten
# Columns and datatype in array


# ToDo:
# Remove column id
# entity embedding
# handle deprecation
# beautiful code and var names

# columns: anzahl fahrzeuge, datum

from __future__ import absolute_import      #
from __future__ import division             #
from __future__ import print_function       #

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder




######################################## Prepare data ########################################

# Not important
header = ['ID', 'Strassenklasse', 'Unfalldatum', 'Alter', 'Unfallklasse',
          'Unfallschwere', 'Lichtverhaeltnisse', 'VerletztePersonen',
          'AnzahlFahrzeuge', 'Bodenbeschaffenheit', 'Geschlecht', 'Zeit',
          'Fahrzeugtyp', 'Wetterlage']

headerTest = ['ID', 'Strassenklasse', 'Unfalldatum', 'Alter', 'Unfallklasse',
          'Lichtverhaeltnisse', 'VerletztePersonen',
          'AnzahlFahrzeuge', 'Bodenbeschaffenheit', 'Geschlecht', 'Zeit',
          'Fahrzeugtyp', 'Wetterlage']


dtypes = {
    'Strassenklasse': str,
    'Unfalldatum': str,
    'Alter': np.int32,
    'Unfallklasse': str,
    'Unfallschwere': np.int32,
    'Lichtverhaeltnisse': str,
    'VerletztePersonen': np.int32,
    'AnzahlFahrzeuge': np.int32,
    'Bodenbeschaffenheit': str,
    'Geschlecht': str,
    'Zeit': np.int32,
    'Fahrzeugtyp': str,
    'Wetterlage': str
}

feature_columns = [
    tf.feature_column.categorical_column_with_vocabulary_list('Strassenklasse', vocabulary_list=['Bundesstrasse', 'nicht klassifiziert', 'Landesstrasse', 'Kraftfahrzeugstrasse', 'Autobahn']),
    tf.feature_column.numeric_column('Alter'),
    tf.feature_column.categorical_column_with_vocabulary_list('Unfallklasse', vocabulary_list=['Fahrer', 'Passagier', 'Fussgänger']),
    #tf.feature_column.numeric_column('Unfallschwere'),
    tf.feature_column.categorical_column_with_vocabulary_list('Lichtverhaeltnisse', vocabulary_list=['Tageslicht: Strassenbeleuchtung vorhanden', 'Dunkelheit: Strassenbeleuchtung vorhanden und beleuchtet', 'Dunkelheit: Strassenbeleuchtung vorhanden und nicht beleuchtet', 'Dunkelheit: Strassenbeleuchtung unbekannt', 'Dunkelheit: keine Strassenbeleuchtung']),
    tf.feature_column.numeric_column('VerletztePersonen'),
    tf.feature_column.numeric_column('AnzahlFahrzeuge'),
    tf.feature_column.categorical_column_with_vocabulary_list('Bodenbeschaffenheit', vocabulary_list=['nass / feucht', 'trocken', 'Frost / Eis', 'Frost/ Ice', 'Schnee']),
    tf.feature_column.categorical_column_with_vocabulary_list('Geschlecht', vocabulary_list=['männlich', 'weiblich']),
    tf.feature_column.numeric_column('Zeit'),
    tf.feature_column.categorical_column_with_vocabulary_list('Fahrzeugtyp', vocabulary_list=['Auto', 'Bus', 'Fahrrad', 'Taxi', 'Mottorrad (50cc)', 'Kleinbus', 'Transporter', 'Mottorrad (125cc)' 'Mottorrad (500cc)', 'Andere', 'LKW bis 7.5t']),
    tf.feature_column.categorical_column_with_vocabulary_list('Wetterlage', vocabulary_list=['Gut', 'Schnee' 'Andere', 'Regen (starker Wind)', 'Unbekannt', 'Gut (starker Wind)', 'Regen']),
]

#################### Train data

df_train = pd.read_csv('input/verkehrsunfaelle_train.csv', sep=',', dtype=dtypes, na_values="?", encoding='latin-1')

df_train.columns = header
df_train.drop('ID', axis=1, inplace=True)
df_train.drop('Unfalldatum', axis=1, inplace=True)

num_train_entries = df_train.shape[0]
num_train_features = df_train.shape[1] - 1

#################### Test data

df_test = pd.read_csv('input/verkehrsunfaelle_test.csv', sep=',', dtype=dtypes, na_values="?", encoding='latin-1')

df_test.columns = headerTest
df_test.drop('ID', axis=1, inplace=True)
df_test.drop('Unfalldatum', axis=1, inplace=True)

num_test_entries = df_test.shape[0]
num_test_features = df_test.shape[1] - 1

print(num_train_entries)
print(num_train_features)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_train.head())

def load_data(x_train, x_test, y_name="Unfallschwere", seed=None):

    # Delete rows with unknowns
    x_train.dropna()
    x_test.dropna()

    # Shuffle the data
    np.random.seed(seed)

    # Extract the label from the features DataFrame.
    y_train = x_train.pop(y_name)
    y_test = x_test #.pop(y_name)     # NOT EXISTENT >>>

    return (x_train, y_train), (x_test, y_test)




######################################## Prepare Model ########################################

(train_x, train_y), (test_x, test_y) = load_data(df_train, df_test)

#train_y /= args.price_norm_factor
#test_y /= args.price_norm_factor

# Make input function for training:
#   num_epochs=None -> will cycle through input data forever
#   shuffle=True -> randomize order of input dat
training_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, batch_size=64,
                                                        shuffle=True, num_epochs=None)

# Make input function for evaluation:
# shuffle=False -> do not randomize input data
eval_input_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, batch_size=64, shuffle=False)

model = tf.estimator.LinearRegressor(feature_columns=feature_columns, model_dir="./model")




######################################## Train Model ########################################

############## Train
model.train(input_fn=training_input_fn, steps=20000)

############## Evaluate
# Evaluate how the model performs on data it has not yet seen.
eval_result = model.evaluate(input_fn=eval_input_fn)

# The evaluation returns a Python dictionary. The "average_loss" key holds the
# Mean Squared Error (MSE).
average_loss = eval_result["average_loss"]

# Convert MSE to Root Mean Square Error (RMSE).
print("\n" + 80 * "*")
print("\nRMS error for the test set: ", average_loss)

######################################## Prediction ##########################################

df = train_x[:3]

print(df)
predict_input_fn = tf.estimator.inputs.pandas_input_fn(x=df, shuffle=False)

predict_results = model.predict(input_fn=predict_input_fn)

# Print the prediction results.
print("\nPrediction results:")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    for i, prediction in enumerate(predict_results):
        print("Final prefiction is:", prediction.values())