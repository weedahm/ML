from ikzziML.common import generateData
from ikzziML.common import mappingData
from ikzziML.common import manipulateMissingVal

DB_PATH = 'data/db/db.sqlite3'
SAVE_PATH = 'data/created_csv/patients_'
PATIENTS_X_CSV_PATH = SAVE_PATH + 'X.csv'
PATIENTS_TRAINING_X_CSV_PATH = SAVE_PATH + 'X_Training.csv'
PATIENTS_Y_CSV_PATH = SAVE_PATH + 'Y.csv'
PATIENTS_Y_SET_CSV_PATH = SAVE_PATH + 'Y_set.csv'

def menu():
    print("1. Generate: data & mapping to number")
    print("2. Manipulate: missing value")
    print("3. Manipulate: delete samples of zero prescription")
    try:
        m = int(input("Select: "))
    except ValueError:
        print("Invalid number")
    return m

m = menu()

if(m == 1):
    ##### generate Data (DB to CSV)
    df_raw = generateData.readDB(DB_PATH)
    df_input, df_output, df_output_set = generateData.castToMLData(df_raw)
    generateData.saveToCSV(df_input, df_output, df_output_set, SAVE_PATH)

    ##### patients_X mapping to number 한글, 영어 값 -> 숫자 처리
    df_t = mappingData.loadCSV(PATIENTS_X_CSV_PATH)
    training_df_X = mappingData.mappingToTrainingValues(df_t)
    mappingData.saveCSV(training_df_X, SAVE_PATH)

elif(m == 2):
    ##### manipulate missing value (**HARD CODING**) 복진 0 값 처리
    df_train_X = mappingData.loadCSV(PATIENTS_TRAINING_X_CSV_PATH)
    manip_df_train_X = manipulateMissingVal.totalMani(df_train_X)
    mappingData.saveCSV(manip_df_train_X, SAVE_PATH)

elif(m == 3):
    ##### delete samples of Zero Prescription
    df_train_X = mappingData.loadCSV(PATIENTS_TRAINING_X_CSV_PATH)
    df_Y = mappingData.loadCSV(PATIENTS_Y_CSV_PATH)
    df_Y_set = mappingData.loadCSV(PATIENTS_Y_SET_CSV_PATH)
    manipulateMissingVal.deleteZeroPres(df_train_X, df_Y, df_Y_set, SAVE_PATH)
