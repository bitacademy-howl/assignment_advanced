import pandas as pd
from actions.data_preprocessing import train, test
from actions.settings import Settings

if __name__ == '__main__':
    settings = Settings()

    print(test[pd.isnull(test["Fare"])])

    mean_fare = train["Fare"].mean()
    test.loc[pd.isnull(test["Fare"]), "Fare"] = mean_fare


