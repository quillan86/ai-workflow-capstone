from app.pipeline.pipeline import train, predict_date

if __name__ == "__main__":
#    train("initial_model.db")
    y = predict_date("initial_model.db", "united kingdom", "2019-10-13")
    y = float(y[0])
    print(y)
#    load("initial_model")