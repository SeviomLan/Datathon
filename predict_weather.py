import joblib
from csv import reader

def predict(model, list):
    temp = model.predict(list)
    return temp

def print_values(input, bool):
    sums = sum(input)
    if bool:
        x = "temperatures"
        print("Average temperature for the given months: " + str(sums/len(input)))
    else:
        x = "rains"
        print("Total rain for the given months: " + str(sums))
    print("Predicted " + x + " for the given months: " + str(input))

# def predict_rain(model, list, temps):
#     for count, value in enumerate(temps):
#         list[count].append(value)
#     rain = model.predict(list[0].reshape(-1, 1))
#     print(rain)

def read_input(file):
    with open(file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        list_of_rows = list(csv_reader)
    return list_of_rows

if __name__ == '__main__':
    temp_model = joblib.load('tem_model.pkl')
    rain_model = joblib.load('rain_model_single.pkl')
    months = read_input('input.csv')

    temperatures = predict(temp_model, months)
    print_values(temperatures, True)
    
    rains = predict(rain_model, months)
    print_values(rains, False)