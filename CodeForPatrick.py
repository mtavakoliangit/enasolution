# assumptions in weather_data dictionary:
# 1. format matches
# 2. temperature in Kelvin
# 3. pressure in hPa
# 4. cloud is in percentage 
# 5. data is hourly (difference between dt's = 3600 sec)

weather_data = {"message": "Count: 25", "cod": "200", "city_id": 1, "calctime": 0.005118065, "cnt": 25, 
                "list": [{"dt": 1633071600, 
                          "main": {"temp": 285.27, "feels_like": 283.92, "pressure": 1011, "humidity": 53, "temp_min": 282.77, "temp_max": 286.76}, 
                          "wind": {"speed": 0.89, "deg": 309, "gust": 2.68}, 
                          "clouds": {"all": 100}, 
                          "weather": [{"id": 804, "main": "Clouds", "description": "overcast clouds", "icon": "04n"}]}, 
                         {"dt": 1633075200, 
                          "main": {"temp": 285.17, "feels_like": 283.81, "pressure": 1012, "humidity": 53, "temp_min": 281.66, "temp_max": 286.76}, 
                          "wind": {"speed": 1.34, "deg": 350, "gust": 4.47}, 
                          "clouds": {"all": 100}, 
                          "weather": [{"id": 804, "main": "Clouds", "description": "overcast clouds", "icon": "04n"}]}, 
                         {"dt": 1633078800, 
                          "main": {"temp": 285.09, "feels_like": 283.57, "pressure": 1013, "humidity": 47, "temp_min": 281.11, "temp_max": 286.21}, 
                          "wind": {"speed": 1.34, "deg": 321, "gust": 2.68}, 
                          "clouds": {"all": 100}, 
                          "weather": [{"id": 804, "main": "Clouds", "description": "overcast clouds", "icon": "04n"}]}]}

inside_data = {'reading_temp' : 16,
               'setpoint' : 20,
               'humidity' : 18}

# imports
from tensorflow import keras
from pickle import load
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def Test_Weather_Data_Validity (weather_data):
    valid = True
    for i in range(len(weather_data['list']) - 1):
        if weather_data['list'][i+1]['dt'] - weather_data['list'][i]['dt'] != 3600:
            valid = False
    if not valid:
        print('Wrong weather data format. Make sure the frequency of your weather data is 1 hour.')
    return valid

def Create_InputData_From_Weather_And_Inside (weather_data, inside_data):
    input_data = inside_data
    if Test_Weather_Data_Validity (weather_data):
        input_data['weather'] = weather_data['list']
    return input_data

def Read_Learner (path):
    return keras.models.load_model(path)

def Step (x1, x2):
    return (x2 - x1) / 4

def Define_Columns():
    return ['setpoint', 'humidity', 'pressure_sea', 'wind_speed', 'outside_humidity', 
            'cloud_cover_8', 'dt_reading_and_setpoint', 'dt_outside_and_setpoint']

def Prep_Input_Data (input_data):
    cols = Define_Columns()
    df = pd.DataFrame(columns=cols)
    df['out_temp'] = ''
    if 'weather' in input_data.keys():
        sp = input_data['setpoint']
        humidity = input_data['humidity']
        reading_temp = input_data['reading_temp']
        for i in range(len(input_data['weather']) - 1):
            # hPa to kPa
            ps1 = input_data['weather'][i]['main']['pressure'] / 10
            ps2 = input_data['weather'][i+1]['main']['pressure'] / 10
            # m/sec to km/hr
            wind1 = input_data['weather'][i]['wind']['speed'] * 3.6
            wind2 = input_data['weather'][i+1]['wind']['speed'] * 3.6
            out_humidity1 = input_data['weather'][i]['main']['humidity']
            out_humidity2 = input_data['weather'][i+1]['main']['humidity']
            cloud1 = input_data['weather'][i]['clouds']['all'] / 100
            cloud2 = input_data['weather'][i+1]['clouds']['all'] / 100
            # Kelvin to degC
            outTemp1 = input_data['weather'][i]['main']['temp'] - 273.15
            outTemp2 = input_data['weather'][i+1]['main']['temp'] - 273.15
            for count in range(4):
                df.loc[len(df)] = [sp, humidity, 
                                   ps1 + count * Step(ps1, ps2),
                                   wind1 + count * Step(wind1, wind2),
                                   out_humidity1 + count * Step(out_humidity1, out_humidity2),
                                   cloud1 + count * Step(cloud1, cloud2),
                                   '', '',
                                   outTemp1 + count * Step(outTemp1, outTemp2)]
        df['time'] = ''
        df['time'][0] = str(0) + ' min'
        df['pred_temp'] = ''
        df['pred_temp'][0] = reading_temp
        df['dt_reading_and_setpoint'][0] = reading_temp - sp
        df['dt_outside_and_setpoint'][0] = input_data['weather'][0]['main']['temp'] - 273.15 - sp
    return df

def Predict_Temp (weather_data, inside_data):
    input_data = Create_InputData_From_Weather_And_Inside (weather_data, inside_data)
    df = Prep_Input_Data(input_data)
    model1 = Read_Learner('Models/MoreInputs')
    model2 = Read_Learner('Models/MoreInputsModified')
    cols = Define_Columns()
    for i in range(len(df) - 1):
        sp = df['setpoint'][i]
        humidity = df['humidity'][i]
        ps = df['pressure_sea'][i]
        wind = df['wind_speed'][i]
        out_humid = df['outside_humidity'][i]
        cloud = df['cloud_cover_8'][i]
        outTemp = df['dt_outside_and_setpoint'][i] + sp
        reading = df['pred_temp'][i]
        inputData1 = [sp, humidity, ps, wind, out_humid, cloud, reading - sp, outTemp - sp]
        inputData2 = [humidity, ps, wind, out_humid, cloud, reading - sp, outTemp - sp]
        inputScaled1 = []
        inputScaled2 = []
        for col in cols:
            # load StandardScaler and MinMaxScaler from training 
            norm = load(open('Models/normer_' + col + '.pkl', 'rb'))
            scale = load(open('Models/scaler_' + col + '.pkl', 'rb'))
            index1 = cols.index(col)
            index2 = cols.index(col) - 1
            # normer and scaler have formed based on 2d array (dataframe)
            # convert single point data to 2d format
            data1 = pd.DataFrame([inputData1[index1]], columns=[col])
            inputScaled1.append(norm.transform(scale.transform(data1)))
            # inputScaled is formed as an array of arrays
            # retrieve values in regular list format
            inputScaled1[index1] = inputScaled1[index1][0][0]
            if col != 'setpoint':
                data2 = pd.DataFrame([inputData2[index2]], columns=[col])
                inputScaled2.append(norm.transform(scale.transform(data2)))
                inputScaled2[index2] = inputScaled2[index2][0][0]
        xVal = pd.DataFrame([inputScaled1], columns=cols)
        yVal_pred = pd.DataFrame(model1.predict(xVal), columns=['temp_pred'])
        normToInverseOutput = load(open('Models/normer_temp_15minAhead.pkl', 'rb'))
        scaleToInverseOutput = load(open('Models/scaler_temp_15minAhead.pkl', 'rb'))
        yPred_denorm = normToInverseOutput.inverse_transform(yVal_pred)
        pred_temp1 = scaleToInverseOutput.inverse_transform(yPred_denorm)[0][0]
        if col != 'setpoint':
            modCols = Define_Columns().remove('setpoint')
            xVal = pd.DataFrame([inputScaled2], columns=modCols)
            yVal_pred = pd.DataFrame(model2.predict(xVal), columns=['temp_pred'])
            normToInverseOutput = load(open('Models/normer_dt_15minAhead_setpoint.pkl', 'rb'))
            scaleToInverseOutput = load(open('Models/scaler_dt_15minAhead_setpoint.pkl', 'rb'))
            yPred_denorm = normToInverseOutput.inverse_transform(yVal_pred)
            step = scaleToInverseOutput.inverse_transform(yPred_denorm)[0][0]
            pred_temp2 = sp + step
        else:
            pred_temp2 = pred_temp1
        pred_temp = (pred_temp1 + pred_temp2) / 2
        df['pred_temp'][i+1] = pred_temp
        df['dt_reading_and_setpoint'][i+1] = pred_temp - sp
        df['dt_outside_and_setpoint'][i+1] = df['out_temp'][i+1] - sp
        df['time'][i+1] = str((i + 1) * 15) + ' min'
    return df[['time', 'pred_temp']]