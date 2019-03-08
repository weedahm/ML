import generateData
import mappingData

DB_PATH = 'db/db.sqlite3'
PATIENTS_X_CSV_PATH = 'created_csv/patients_X_new.csv'

##### generate Data (DB to CSV)
'''
df_raw = generateData.readDB(DB_PATH)
df_input, df_output, df_output_set = generateData.castToMLData(df_raw)
generateData.saveToCSV(df_input, df_output, df_output_set)
'''

##### patients_X mapping to number
df_X = mappingData.loadCSV(PATIENTS_X_CSV_PATH)
training_df_X = mappingData.mappingToTrainingValues(df_X)
mappingData.saveCSV(training_df_X)